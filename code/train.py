import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from transformers import BertTokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.losses import MultilabelCategoricalCrossentropy
from bert4torch.layers import GlobalPointer, CRF
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

maxlen = 256
hidden_size = 768
lstm_num_layers = 1
batch_size = 16
threshold = -1.5
trainfile = "datasets/kp20k/train.txt"
validfile = "datasets/kp20k/valid.txt"
# gp
gp_categories_label2id = {"CW": 0, "SW": 1, "O": 2}
gp_categories_id2label = dict((value, key) for key,value in gp_categories_label2id.items())
head_num = len(gp_categories_label2id)  # 即heads
head_size = 64
# crf
crf_categories = ['O', 'B-CW', 'I-CW', 'E-CW', 'S-SW']
crf_categories_id2label = {i: k for i, k in enumerate(crf_categories)}
crf_categories_label2id = {k: i for i, k in enumerate(crf_categories)}

# BERT base
config_path = 'pretrain_models/bert-base-uncased/config.json'
checkpoint_path = 'pretrain_models/bert-base-uncased/bert4torch_pytorch_model.bin'
dict_path = 'pretrain_models/bert-base-uncased/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 固定seed
seed_everything(42)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        data = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            
            for l in f.split('\n\n'):
                if not l:
                    continue
                text, label = [], []
                
                for i, c in enumerate(l.split('\n')):
                    if c[0:10] == "docnumber:":
                        # print(c)
                        continue
                    char, flag = c.split(' ')
                    text.append(char)
                    if flag[0] == 'B':
                        label.append([i, i, flag[2:]])
                    elif flag[0] == 'E':
                        label[-1][1] = i
                    elif flag[0] == 'S':
                        label.append([i, i, flag[2:]])
                if text:
                    data.append((text, label))  # text为["token", ...], label为[[start, end, entity], ...]
        return data


# 建立分词器
tokenizer = BertTokenizer.from_pretrained('pretrain_models/bert-base-uncased')
def collate_fn(batch):
    batch_token_ids = []  # (batch_size, seq_len)
    batch_labels_gp = []  # (batch_size, 2, seq_len, seq_len)
    batch_labels_crf = []
    for i, (text, text_labels) in enumerate(batch):
        token_ids = tokenizer.encode_plus(text, add_special_tokens = True, max_length=maxlen, truncation=True)["input_ids"]

        labels_gp = np.zeros((head_num, maxlen, maxlen))
        labels_crf = np.zeros(len(token_ids))
        for start, end, label in text_labels:
            if start < maxlen-3 and end < maxlen-3 and label == "CW":
                labels_crf[start + 1] = crf_categories_label2id['B-'+label]
                labels_crf[start + 2:end + 1] = crf_categories_label2id['I-'+label]
                labels_crf[end + 1] = crf_categories_label2id['E-'+label]
                label = gp_categories_label2id[label]
                labels_gp[label, start + 1, end + 1] = 1

            if start < maxlen-2 and end < maxlen-2 and label == "SW":
                labels_crf[start + 1] = crf_categories_label2id['S-'+label]
                label = gp_categories_label2id[label]
                labels_gp[label, start + 1, end + 1] = 1

            if start < maxlen-2 and end < maxlen-2 and label == "O":
                labels_crf[start + 1] = crf_categories_label2id["O"]
                label = gp_categories_label2id[label]
                labels_gp[label, start + 1, end + 1] = 1

        batch_token_ids.append(token_ids)
        batch_labels_gp.append(labels_gp[:, :len(token_ids), :len(token_ids)])
        batch_labels_crf.append(labels_crf)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels_gp = torch.tensor(sequence_padding(batch_labels_gp, seq_dims=3), dtype=torch.long, device=device)
    batch_labels_crf = torch.tensor(sequence_padding(batch_labels_crf), dtype=torch.long, device=device)

    return batch_token_ids, [batch_labels_gp, batch_labels_crf]


# 转换数据集
train_dataloader = DataLoader(MyDataset(trainfile), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset(validfile), batch_size=batch_size, collate_fn=collate_fn) 


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='bert', segment_vocab_size=0)  # output_all_encoded_layers=True
        self.global_pointer = GlobalPointer(hidden_size=hidden_size, heads=head_num, head_size=head_size)
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size//2, num_layers=lstm_num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(hidden_size, len(crf_categories))  # 包含首尾
        self.fc2 = nn.Linear(len(crf_categories), len(crf_categories)) 
        self.crf = CRF(len(crf_categories))

    def forward(self, token_ids):
        attention_mask = token_ids.gt(0).long()  # [btz, seq_len]
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hidden_size]
        sequence_output, _ = self.lstm(sequence_output)  # [btz, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)  # [btz, seq_len, hidden_size]
        crf_emission_score = self.fc1(sequence_output)  # [btz, seq_len, 5]
        logit = self.global_pointer(sequence_output, token_ids.gt(0).long())  # [btz, heads, seq_len, seq_len]
        gp_emission_score = self.trans_logit2emission_score(logit, attention_mask)  # [btz, seq_len, 5]
        gp_emission_score_normalized = torch.softmax(gp_emission_score, dim=-1)  # [btz, seq_len, 5]
        emission_score = torch.mul(gp_emission_score_normalized, crf_emission_score) + crf_emission_score  # [btz, seq_len, 5]
        return [logit, emission_score, attention_mask]

    def trans_logit2emission_score(self, logit, attention_mask):
        # 将gp的打分结果转化为emission_score
        btz, heads, seq_len, seq_len = logit.size()

        # 将第0个词的I和E填为-float("inf")
        b_emission_score, _ = torch.max(logit[:,0,0,1:], dim=-1, keepdim=True)  # [btz, 1]
        i_emission_score = torch.ones((btz, 1), dtype=torch.float, device=device) * -float("inf")
        e_emission_score = torch.ones((btz, 1), dtype=torch.float, device=device) * -float("inf")
        for i in range(1, seq_len):
            if 0 < i < seq_len-1: 
                b_values, _ = torch.max(logit[:,0,i,i+1:], dim=-1, keepdim=True)  # [btz, 1]
                i_values, _ = torch.max(logit[:,0,0:i,i+1:], dim=-1, keepdim=True)  # [btz, seq_len, 1]
                i_values, _ = torch.max(i_values, dim=-2)  # [btz, 1]
                e_values, _ = torch.max(logit[:,0,0:i,i], dim=-1, keepdim=True)  # [btz, 1]
            else:  # 将最后一个单词的B和I填为-float("inf")
                b_values= torch.ones((btz, 1), dtype=torch.float, device=device) * -float("inf")
                # torch.randn((btz, 1), device=device)
                i_values= torch.ones((btz, 1), dtype=torch.float, device=device) * -float("inf")
                e_values, _ = torch.max(logit[:,0,0:i,i], dim=-1, keepdim=True)  # [btz, 1]

            b_emission_score = torch.cat([b_emission_score, b_values], dim=-1)
            i_emission_score = torch.cat([i_emission_score, i_values], dim=-1)
            e_emission_score = torch.cat([e_emission_score, e_values], dim=-1)

        b_emission_score = b_emission_score.unsqueeze(-1)  # [btz, seq_len, 1]
        i_emission_score = i_emission_score.unsqueeze(-1)  # [btz, seq_len, 1]
        e_emission_score = e_emission_score.unsqueeze(-1)  # [btz, seq_len, 1]
        
        s_emission_score = logit[:,1,:,:]  # [btz, seq_len, seq_len]
        s_emission_score = torch.diagonal(s_emission_score, dim1=-2, dim2=-1).unsqueeze(-1)  # [btz, seq_len, 1]
        
        o_emission_score = logit[:,2,:,:]  # [btz, seq_len, seq_len]
        o_emission_score = torch.diagonal(o_emission_score, dim1=-2, dim2=-1).unsqueeze(-1)  # [btz, seq_len, 1]
        
        gp_emission_score = torch.cat([o_emission_score, 
                                       b_emission_score, 
                                       i_emission_score, 
                                       e_emission_score, 
                                       s_emission_score], dim=-1)

        # mask掉padding的部分
        mask = 1 - attention_mask.unsqueeze(-1)
        gp_emission_score = gp_emission_score.masked_fill(mask.bool(), value=0.0)

        return gp_emission_score  # [btz, seq_len, 5]


model = Model().to(device)


class Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mlcce = MultilabelCategoricalCrossentropy()
    def forward(self, y_pred, y_true):
        crf_pred = y_pred[1:]  # [emission_score, attention_mask]
        crf_true = y_true[1]  # [btz, seq_len]
        crf_loss = model.crf(*crf_pred, crf_true)

        gp_pred = y_pred[0]  # logit: [btz, heads, seq_len, seq_len]
        gp_true = y_true[0]
        gp_pred = gp_pred.view(gp_pred.shape[0]*gp_pred.shape[1], -1)  # [btz*head_num, seq_len*seq_len]
        gp_true = gp_true.view(gp_true.shape[0]*gp_true.shape[1], -1)  # [btz*head_num, seq_len*seq_len]
        gp_loss = self.mlcce.forward(gp_pred, gp_true)
        gp_loss = self.mlcce.forward(gp_pred, gp_true)

        return gp_loss + crf_loss


loss=Loss()
learning_rate = 2e-5
optimizer=optim.Adam(model.parameters(), lr=learning_rate)
model.compile(loss=loss, optimizer=optimizer)


def evaluate(data, threshold = threshold):
    tp_sk, tpfp_sk, tpfn_sk, support_sk = 0, 1e-10, 1e-10, 0
    tp_ck, tpfp_ck, tpfn_ck, support_ck = 0, 1e-10, 1e-10, 0
    for batch_token_ids, batch_labels in tqdm(data, ncols=100):
        batch_labels_gp, batch_labels_crf = batch_labels
        logit, emission_score, attention_mask = model.predict(batch_token_ids)  # logit:[btz, heads, seq_len, seq_len]
        best_path = model.crf.decode(emission_score, attention_mask)  # [btz, seq_len]

        R1 = set()  # {(样本id, start, end, 实体类型)}
        for i, score in enumerate(logit):
            for l, start, end in zip(*np.where(score.cpu() > threshold)):  # 打分在threshold以上的即为预测出来的实体
                R1.add((i, start, end, gp_categories_id2label[l]))   
        R2 = trans_entity2tuple(best_path)
        R = R1 | R2

        T1 = set()
        for i, score in enumerate(batch_labels_gp):
            for l, start, end in zip(*np.where(score.cpu() == 1)):  # 打分在0以上(其实就是打分为1)的为真实的实体
                T1.add((i, start, end, gp_categories_id2label[l]))

        T2 = trans_entity2tuple(batch_labels_crf)
        T = T1 | T2

        R_sk, R_ck = divide2sk_ck(R)
        T_sk, T_ck = divide2sk_ck(T)

        # sk
        tp_sk +=  len(R_sk & T_sk)  # TP, 预测正确的正样本
        tpfp_sk +=  len(R_sk)  # TP+FP, 预测的正样本
        tpfn_sk +=  len(T_sk)  # TP+FN, 真实的正样本
        support_sk += len(T_sk)

        # ck
        tp_ck +=  len(R_ck & T_ck)  # TP, 预测正确的正样本
        tpfp_ck +=  len(R_ck)  # TP+FP, 预测的正样本
        tpfn_ck +=  len(T_ck)  # TP+FN, 真实的正样本
        support_ck += len(T_ck)
    
    micro_p = (tp_sk + tp_ck) / (tpfp_ck + tpfp_sk)
    micro_r = (tp_sk + tp_ck) / (tpfn_ck + tpfn_sk)
    micro_f1 = 2 * (tp_sk + tp_ck) / (tpfp_ck + tpfp_sk + tpfn_ck + tpfn_sk)

    precision_sk, recall_sk, f1_sk = tp_sk / tpfp_sk, tp_sk / tpfn_sk, 2 * tp_sk / (tpfp_sk + tpfn_sk)
    precision_ck, recall_ck, f1_ck = tp_ck / tpfp_ck, tp_ck / tpfn_ck, 2 * tp_ck / (tpfp_ck + tpfn_ck)
    macro_p = (precision_sk + precision_ck) / 2
    macro_r = (recall_sk + recall_ck) / 2
    macro_f1 = (f1_sk + f1_ck) / 2
    
    support = support_ck + support_sk
    print("\tprecision\trecall\t\tf1\t\tsupport")
    print("sk\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}".format(precision_sk, recall_sk, f1_sk, support_sk))
    print("ck\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}".format(precision_ck, recall_ck, f1_ck, support_ck))
    print("micro\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}".format(micro_p, micro_r, micro_f1, support))
    print("macro\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}".format(macro_p, macro_r, macro_f1, support))


    return micro_f1


def trans_entity2tuple(scores):
    '''把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    '''
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):  # [0, 1, 2, 3, 0, 4, 0, 1, 3, 0, ...]
            flag_tag = crf_categories_id2label[item.item()]
            if flag_tag.startswith('B-'):  # B
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif flag_tag.startswith('S-'):  # S
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and (flag_tag[2:]==entity_ids[-1][-1]):  # I
                entity_ids[-1][-2] = j
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('E-') and (flag_tag[2:]==entity_ids[-1][-1]):  # E
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])

        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids


def divide2sk_ck(res):
    res_sk = set()
    res_ck = set()
    for i in res:
        if i[2] == i[1]:
            res_sk.add(i)
        else:
            res_ck.add(i)
    return res_sk, res_ck


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        super().__init__()
        self.best_val_f1 = 0.0

    def on_epoch_end(self, steps, epoch, logs=None):
        micro_f1 = evaluate(valid_dataloader)
        if micro_f1 > self.best_val_f1:
            self.best_val_f1 = micro_f1
            model.save_weights('checkpoints/paecrf'+str(hidden_size)+str(num_layers)+'.pt')
        print(self.best_val_f1)


if __name__ == '__main__':
    evaluator = Evaluator()
    # model.load_weights('checkpoints/paecrf'+str(hidden_size)+str(num_layers)+'.pt')
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])

