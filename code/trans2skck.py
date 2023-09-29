import argparse
import os

def trans2skck(path, parent_dir, file_name):
    with open(os.path.join(path, parent_dir, file_name), "r") as f:
        kps = [i.strip() for i in f.readlines()]  # ["grid servic discoveri;discoveri", ...]

    sk_file = open(os.path.join(path, parent_dir, "simplekeyword.txt"), "w+")
    ck_file = open(os.path.join(path, parent_dir, "complexkeyword.txt"), "w+")
    for kp in kps:
        sk = []
        ck = []
        kp = kp.split(";")  # ["grid servic discoveri", "discoveri", ...]
        for kw in kp:
            kw = kw.strip()
            k = kw.split(" ")  # ["grid", "servic", "discoveri"]
            if len(k) == 1:
                sk.append(kw)  # "discoveri"
            elif len(k) > 1:
                ck.append(kw)  # "grid servic discoveri"
        if file_name == "predictions.txt":
            # sk.append("<peos>")  # 如果原结果中带<peos>就注释掉这行
            ck.append("<peos>")
        sk_str = ";".join(sk)
        ck_str = ";".join(ck)  # "grid servic discoveri;"
        sk_file.write(sk_str + "\n")
        ck_file.write(ck_str + "\n")

    sk_file.close()
    ck_file.close()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trans2skck.py')
    parser.add_argument('-dataset_directorys', nargs='+', type=str, default=[''], help='dataset_directorys')
    parser.add_argument('-model_names', nargs='+', type=str, default=[''], help='model_names')
    args = parser.parse_args()
    print(args)
    path = "datasets"
    dataset_directorys = args.dataset_directorys
    model_names = args.model_names

    for dataset in dataset_directorys:
        for model_name in model_names:
            parent_dir = os.path.join(dataset, model_name)
            trans2skck(path, parent_dir, "predictions.txt")