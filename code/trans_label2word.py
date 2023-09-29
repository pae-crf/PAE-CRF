from bert4torch.snippets import ListDataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trans_label2word.py')
    parser.add_argument('-dataset_directorys', nargs='+', type=str, default=[''], help='dataset_directorys')
    parser.add_argument('-model_names', nargs='+', type=str, default=[''], help='model_names')
    parser.add_argument('-batch_size', type=int, default=16, help='batch_szie')
    args = parser.parse_args()
    print(args)
    sentence_transformers_path = 'pretrain_models/all-mpnet-base-v1'
    model = SentenceTransformer(sentence_transformers_path, device='cuda')
    dir = "datasets"
    dataset_directorys = args.dataset_directorys
    model_names = args.model_names
    batch_size = args.batch_size

    for model_name in model_names:
        for dataset in dataset_directorys:
            dataset_path = os.path.join(dir, dataset, "test.txt")
            data = MyDataset(dataset_path)

            result_label_file_path = os.path.join(dir, dataset, model_name, "result_label.txt")
            with open(result_label_file_path, "r") as f:
                res = [eval(i.strip()) for i in f.readlines()]

            # 将一行batch_size条数据转化为一行1条数据
            with open(result_label_file_path, "w+") as f:
                for i in range(len(res)):
                    for j in range(batch_size):
                        if i * batch_size + j < len(data):
                            r = {}
                            r["All"] = {kp for kp in res[i]["All"] if kp[0] == j}
                            r["SK"] = {kp for kp in res[i]["SK"] if kp[0] == j}
                            r["CK"] = {kp for kp in res[i]["CK"] if kp[0] == j}
                            f.write(str(r) + "\n")

            with open(result_label_file_path, "r") as f:
                res = [eval(i.strip()) for i in f.readlines()]

            # 将(0, 8, 11, 'CW')转为关键词，存入predictions.txt中
            result_word_file_path = os.path.join(dir, dataset, model_name, "predictions.txt")
            with open(result_word_file_path, "w+") as f:
                for i in tqdm(range(len(data)), ncols=100):
                    text = data[i][0]
                    all_result = res[i]["All"]
                    kp = []
                    if all_result:
                        for k in all_result:
                            start = k[1]  # 关键词的开始坐标
                            end = k[2]  # 关键词的结束坐标
                            kp.append(text[start-1:end])  # [["a", "b"], ...]
                        text = " ".join(text)
                        kp = [" ".join(j) for j in kp]  # ["a b", ...]
                        """
                        将每个关键词也当作一个句子，
                        使用sentence-transformer编码文本和所有关键词的句向量并计算余弦相似度，
                        以之作为关键词地重要程度排序
                        """
                        sentences = []
                        sentences.append(text)
                        sentences.extend(kp)
                        embeddings = model.encode(sentences)
                        sim = util.cos_sim([embeddings[0]], embeddings[1:])
                        kp_rank = [(kp, rank) for kp, rank in zip(*(sentences[1:], sim.tolist()[0]))]
                        kp_rank = sorted(kp_rank, key=lambda x : x[1], reverse=True)
                        kp = [j[0] for j in kp_rank]

                        f.write(";".join(kp) + ";<peos>"+ "\n")
                    else:
                        f.write("<peos>"+ "\n")
