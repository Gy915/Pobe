import json

import pandas as pd
import os
import yaml
import argparse
import time
import numpy as np

import shutil


def change_to_gpt_file(source_file, dest_file, len_range=None):
    if len_range is None:
        len_range = [0, -1]

    total_len = 0
    with open(source_file, 'r') as f:
        for line in f:
            total_len += 1
    f.close()
    begin_pos = len_range[0]
    if len_range[1] <= 0:
        end_pos = total_len - len_range[1]
    else:
        end_pos = len_range[1]
    print("begin:", begin_pos, "end:", end_pos)
    cnt_len = 0
    total = end_pos - begin_pos + 1
    res_str = ""
    with open(source_file, 'r') as f:
        cnt = 0
        for line in f:

            if begin_pos <= cnt <= end_pos:
                label, text = line.split(' ', 1)[0], line.split(' ', 1)[1]
                res_str += "<|endoftext|>" + text.replace("\n", "") + "\n"
                cnt_len += len(text.split(" "))
            cnt += 1
    with open(dest_file, 'w') as f1:
        f1.write(res_str)
    print("cnt_num:", cnt_len)
    print("avg:", cnt_len / total)


def change_news_to_gpt_file(s_f, id_class, d_f):
    res_str = ""
    cnt = 0
    with open(s_f, 'r') as f:
        line = f.readline()
        while line:
            try:
                label, text = line.split(' ', 1)[0], line.split(' ', 1)[1]
            except Exception as e:
                pass
            else:
                if label not in id_class:
                    res_str += "<|endoftext|>" + text
            finally:
                line = f.readline()
    with open(d_f, 'w') as f:
        f.write(res_str)


#
def change_clinc_to_gpt(s_f, type, o_path):
    d = pd.read_csv(s_f)
    texts = d["utt"]
    intents = d["intent"]
    res_t = ""
    for i in range(len(texts)):
        text = texts[i]
        intent = intents[i]
        if (type == "test"):
            if (intent != "ood"):
                res_t += intent + " " + text + "\n"
        elif (type == "ood"):
            if (intent == "ood"):
                res_t += intent + " " + text + "\n"
        else:
            res_t += intent + " " + text + "\n"
    test_f = os.path.join(o_path, "clinc_valid_ood.txt".format(type))
    with open(test_f, "w") as f:
        f.write(res_t)

is_share_memory = {"imdb":True, "clinc":False, "rostd_clean":False, "yelp": True, "yelp_large":True
                ,"wos":True}
def load_args(config_path):
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['timestamp'] = str(time.strftime("%Y-%m_%d-%H_%M_%S", time.localtime()))
    parser = argparse.ArgumentParser()
    for key, value in cfg.items():
        parser.add_argument('--{}'.format(key), type=type(value), default=value)
    args = parser.parse_args()

    if args.benchmark.startswith("snips"):
        args.sent_num = 200
        args.is_share_memory = True
    if args.benchmark == "clinc":
        args.sent_num = 200
    if args.benchmark == "rostd_clean":
        args.sent_num = 200
    if args.benchmark == "imdb":
        args.sent_num = 300
    if args.benchmark == "yelp":
        args.sent_num = 300
    if args.benchmark == "yelp_large":
        args.sent_num = 300
    if args.benchmark == "wos":
        args.sent_num = 300
    args.is_share_memory = is_share_memory

    # if args.type == "record" or "special_record":
    #     args.token_num_path = os.path.join(args.database_path_prefix,
    #                                        "analy/{}_{}_token_num.json".format(args.benchmark, args.model_type))
    #     if args.data_root != "None":
    #         args.token_num_path = os.path.join(args.data_root, args.token_num_path)

    # if args.data_root != "None":
    #     args.index_path = os.path.join(args.data_root, args.index_path)
    #     args.tokens_path = os.path.join(args.data_root, args.tokens_path)
    #     if args.model_path.startswith(".") is False:
    #         args.model_path = os.path.join(args.data_root, args.model_path)
        # args.database_path = os.path.join(args.data_root, args.database_path)

    return args


def generate_config_json(args, **kwargs):
    config_res = {}
    ## common info
    args_dict = vars(args)
    for item in args_dict.items():
        if item[0] not in ["is_share_memory", "gpu_model", "gpu_knn_list"]:
            config_res[item[0]] = item[1]

    ## addition info
    config_res.update(kwargs)
    return config_res


def convert_snips_to_gpt(snips_path, dest_dir, type):
    data = pd.read_csv(snips_path)
    if type == "train":
        raw_texts = data["text"].tolist()
        res_str = ""
        for text in raw_texts:
            res_str += "<|endoftext|>" + text.strip() + "\n"
        with open(dest_dir + "_train_gpt.txt", 'w') as f:
            f.write(res_str)
    if type == "test":
        raw_texts = data["text"].tolist()
        raw_labels = data["is_ood"]
        res_test = ""
        res_ood = ""
        for text, label in zip(raw_texts, raw_labels):
            if label == 0:
                res_test += "<|endoftext|>" + text.strip() + "\n"
            if label == 1:
                res_ood += "<|endoftext|>" + text.strip() + "\n"
        with open(dest_dir + "_test_gpt.txt", 'w') as f:
            f.write(res_test)
        with open(dest_dir + "_ood_gpt.txt", 'w') as f:
            f.write(res_ood)
    if type == "val":
        raw_texts = data["text"].tolist()
        raw_labels = data["is_ood"]
        res_test = ""
        for text, label in zip(raw_texts, raw_labels):
            if label == 0:
                res_test += "<|endoftext|>" + text.strip() + "\n"
        with open(dest_dir + "_valid_gpt.txt", 'w') as f:
            f.write(res_test)

def convert_clinc_to_gpt():
    def change_to_gpt(path):
        res = ""
        with open(path, 'r') as f:
            lines = f.readlines()
            for text in lines:
                res += "<|endoftext|>" + text.strip() + "\n"
        return res
    path = "./dataset/clinc/clinc_{}_gpt.txt"
    for type in ["train", "valid", "test", "ood"]:
        res = change_to_gpt(path.format(type))
        with open(path, 'w') as f:
            f.write(res)


def pprint_config(args):
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)
    for item in args_dict.items():
        try:
            print("{:<16}: {}".format(item[0], item[1]))
        except Exception:
            print("error:", item[0])
    print("*" * 30)


'''
Args:
    res: np.array
    args: Namespace
    kwargs: {"file_name":...}
    
'''


def save_res_to_local(res, args, **kwargs):
    save_path_new = os.path.join(args.save_path,
                                 "{}-{}-{}-{}-{}".format(kwargs["file_name"],
                                                         args.model_type,
                                                         args.type,
                                                         args.benchmark,
                                                         args.timestamp))
    if res is not None:
        if ~os.path.exists(save_path_new):
            print("make dir:", save_path_new)
            os.mkdir(save_path_new)
        try:
            config_json = generate_config_json(args, **kwargs)
            np.save(os.path.join(save_path_new, "res.npy"), res)
            json.dump(config_json, open(os.path.join(save_path_new, "config.json"), 'w'))
        except Exception as e:
            print("save error:", e)
            print("delete dir:", save_path_new)
            shutil.rmtree(save_path_new)
        else:
            print("save res to:", save_path_new)


def write_to_file(x, y, path):
    res = ""
    for i in range(len(x)):
        data = x[i]
        label = y[i]
        text = str(label) + " " + data.replace("\n", "") + "\n"
        res += text
    with open(path, 'w') as f:
        f.write(res)


import sklearn.metrics as metrics
def get_auc(y, pred):
    # y: ood is 1，ind is 0
    # pred: ood is larger
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    fpr95 = 1000  # init
    auroc = metrics.auc(fpr, tpr)
    for i in range(len(tpr)):
        if tpr[i] >= 0.95:
            fpr95 = fpr[i]
            break
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred, pos_label=1)
    aupr_out = metrics.auc(recall, precision)

    pred = [-1 * one for one in pred]
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred, pos_label=0)
    aupr_in = metrics.auc(recall, precision)

    return {"auroc": auroc, "fpr95": fpr95, "aupr_out": aupr_out, "aupr_in": aupr_in}






if __name__ == '__main__':
    convert_clinc_to_gpt()
    # s_prefix = "./dataset/SNIPS/"
    # splits = [0, 1, 2, 3, 4]
    # types = ["train", "test", "val"]
    # d_prefix = "./dataset/snips_75"
    # for split in splits:
    #     for type in types:
    #         snips_path = s_prefix + f"snips_{type}_75_{split}.csv"
    #         d_dir = d_prefix + f"_{split}" + f"/snips_75_{split}"
    #         convert_snips_to_gpt(snips_path, d_dir, type)
    # dataset = load_dataset("glue", "sst2")
    # str = ""
    # dataset = dataset["validation"]
    # for i in range(int(len(dataset))):
    #     data = dataset[i]
    #     text = f'{data["sentence"]}'
    #     label = f'{data["label"]}'
    #
    #     str += label + " " + text + "\n"
    # with open("./dataset/sst2/sst2_valid.txt", 'w') as f:
    #     f.write(str)
    #load_args("./config/2080ti2.yaml")
    # change_to_gpt_file("./dataset/wos/wos_train.txt", "./dataset/wos/wos_train_gpt.txt")
    # change_to_gpt_file("./dataset/wos/wos_test.txt", "./dataset/wos/wos_test_gpt.txt")
    # change_to_gpt_file("./dataset/wos/wos_valid.txt", "./dataset/wos/wos_valid_gpt.txt")
    # change_to_gpt_file("./dataset/wos/wos_ood.txt", "./dataset/wos/wos_ood_gpt.txt")

    # args = load_args("./config/2080ti2.yaml")
    # pprint_config(args)
    # type = "test"

    dataset = ["wos","imdb"]

    for d in dataset:
        for split in ["train", "test", "val"]:
            total_len = 0
            word_len = 0
            with open(f"/data1/gaoy/dataset/{d}/{split}.txt", 'r') as f:
                for line in f:
                    word_len += len(line.split(" "))
                    total_len += 1
                    if total_len == 10000 and split == "val":
                        break
            print(f"{d} {split}", word_len, total_len, word_len/total_len)
    # token_path = "./database/imdb_gpt_tokens_final.npy"
    # r = np.load(token_path)
    # print(len(r))
    # change_news_to_gpt_file("./dataset/news/news_{}.txt".format(type), ['0', '1', '2', '3', '4'], "./dataset/news/news_ood_gpt.txt")
    # change_to_gpt_file("./dataset/yelp/yelp_train", "./dataset/yelp/yelp_train_gpt_25000.txt", [0, 24999])
    # change_to_gpt_file("./dataset/yelp/yelp_train", "./dataset/yelp/yelp_valid_gpt_5000.txt", [25000, 34999])
    # 
    # change_to_gpt_file("./dataset/yelp/yelp_train", "./dataset/yelp/yelp_train_gpt_40000.txt", [0, 41999])
    # change_to_gpt_file("./dataset/yelp/yelp_train", "./dataset/yelp/yelp_valid_gpt_10000.txt", [42000, 51999])
    # save_res_to_local(np.zeros(1), args, file_name= "test", add=123)
    # filepath = "./ckpts/gpt2-imdb-batch4-gpu4/training_args.bin"
    # binfile = open(filepath, 'rb')  # 打开二进制文件
    # size = os.path.getsize(filepath)  # 获得文件大小
    # for i in range(size):
    #     data = binfile.read(1)  # 每次输出一个字节
    #     print(data)
    # binfile.close()

    # change_to_gpt_file("./dataset/imdb/imdb_test", "./dataset/imdb/imdb_test_gpt_1.txt", 0)
    # change_to_gpt_file("./dataset/yelp/yelp_test", "./dataset/yelp/yelp_test_gpt_100.txt", 99)
    # path = "./dataset/rostd_clean/"
    # change_to_gpt_file("./dataset/imdb/imdb_valid.txt", "./dataset/imdb/imdb_valid_gpt.txt")
    # change_to_gpt_file("./dataset/imdb/imdb_train.txt", "./dataset/imdb/imdb_train_gpt.txt")
    # change_to_gpt_file("./dataset/imdb/imdb_test.txt", "./dataset/imdb/imdb_test_gpt.txt")
    # change_to_gpt_file("./dataset/imdb/imdb_valid.txt", "./dataset/imdb/imdb_valid_gpt_100.txt", 100)
    # change_to_gpt_file("./dataset/imdb/imdb_train.txt", "./dataset/imdb/imdb_train_gpt_100.txt", 100)
    # change_to_gpt_file("./dataset/imdb/imdb_test.txt", "./dataset/imdb/imdb_test_gpt_100.txt", 100)
    #
    # change_to_gpt_file("./dataset/yelp/yelp_train.txt", "./dataset/yelp/yelp_train_gpt_25000.txt", 25000)
    # change_to_gpt_file("./dataset/clinc/clinc_valid_ood.txt", "./dataset/clinc/clinc_valid_ood_gpt.txt")
    #
    # change_clinc_to_gpt(os.path.join(path, "{}.csv".format("test")), "ood", path)
    # change_clinc_to_gpt(os.path.join(path, "{}.csv".format("test")), "test", path)
    # change_clinc_to_gpt(os.path.join(path, "{}.csv".format("valid")), "valid", path)
    # change_clinc_to_gpt(os.path.join(path, "{}.csv".format("train")), "train", path)
    # types = ["train", "test", "valid", "ood"]
    # data_prefix = "./dataset/clinc/"
    # for type in types:
    #     change_to_gpt_file(data_prefix+type + ".txt", data_prefix + type + "_gpt.txt")
    #     change_to_gpt_file(data_prefix + type + ".txt", data_prefix + type + "_gpt_100.txt", 100)
    # change_clinc_to_gpt(data_prefix + "valid_with_ood.csv", "ood", data_prefix)
