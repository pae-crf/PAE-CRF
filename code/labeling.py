import argparse
from tqdm import tqdm
from nltk.stem import PorterStemmer
ps = PorterStemmer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='labeling.py')
    parser.add_argument('-src_file_path', type=str, default='datasets/copyrnn_datasets/kp20k_sorted/valid_src.txt', help='path of source file')
    parser.add_argument('-trg_file_path', type=str, default='datasets/copyrnn_datasets/kp20k_sorted/valid_trg.txt', help='path of target file')
    parser.add_argument('-sl_src_file_path', type=str, default='datasets/copyrnn_datasets/sequence_labeling/kp20k_valid_src.txt', help='path of source file for sequence label')
    args = parser.parse_args()


    with open(args.src_file_path, "r") as f:
        src = [i.strip() for i in f.readlines()]

    with open(args.trg_file_path, "r") as f:
        trg = [i.strip() for i in f.readlines()]


    for index, (src_l, trg_l) in tqdm(enumerate(zip(src, trg)), ncols= 100):
        src_l_list = src_l.split()
        src_l = [ps.stem(i) for i in src_l.split()]
        trg_l = [[ps.stem(j) for j in i.split()] for i in trg_l.split(";")]

        src_l_len = len(src_l)
        kp = []  # 关键词实体 [(start, end, entity), ...]
        for kw in trg_l:
            kwlen = len(kw)
            for i in range(src_l_len-kwlen):
                cw = src_l[i:kwlen+i]
                if cw == kw and kwlen > 1:
                    kp.append((i, i+kwlen-1, "CW"))
                if cw == kw and kwlen == 1:
                    kp.append((i, i, "SW"))

        label = ["O" for _ in range(src_l_len)]
        for i in kp:
            if label[i[0]] == "O":
                if i[-1] == "SW":
                    label[i[0]] = "S-SW"
                elif i[-1] == "CW":
                    label[i[0]] = "B-CW"
                    label[i[1]] = "E-CW"
                    for j in range(i[1]-i[0]-1):
                        label[i[0]+1+j] = "I-CW"

        # 将当前文本和标签写入文档
        with open(args.sl_src_file_path, "a+", encoding="utf-8") as f:
            f.write("docnumber:{}\n\n".format(index))
            for w, l in zip(src_l_list, label):
                line = w + " " + l + '\n'
                f.write(line)
            f.write('\n')
