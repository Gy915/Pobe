import math
import faiss
import numpy as np
import time
from KNN_gather import *

dim = 768


# output_path = "./database/yelp_test_100_mem.npy"
# query_store = np.memmap(output_path, dtype=np.float32, mode="r", shape=(10000, 769))


# index_flat = faiss.IndexFlatL2(dim)
# print("begin add")
# index_flat.add(database_store[:, 1:].astype('float32'))
# faiss.write_index(index_flat, "./database/imdb_index_300.index")
# print("begin read")
# index = faiss.read_index("./database/imdb_index_300.index")

def get_gpu_index(index_path, **kwargs):
    index = faiss.read_index(index_path)
    print("to gpu")
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    gpu_index = faiss.index_cpu_to_gpus_list(index, gpus = kwargs["gpu_list"], co=co)
    print("load gpu")
    return index, gpu_index




def get_faiss_and_token_index(tokens_path,
                              index_path, **kwargs):
    index_tokens = np.load(tokens_path, allow_pickle=True)  # index-tokenID
    index, gpu_index = get_gpu_index(index_path, **kwargs)
    return gpu_index, index_tokens


def create_index_tokens(args):
    datastore_path = args.database_path
    shape = args.shape
    output_path = args.tokens_path
    print("\n*************create index_tokens*************\n")
    print(" database_path: {}".format(datastore_path))
    database = np.memmap(datastore_path, dtype=np.float32, mode="r",
                         shape=shape)
    pos = find_last_pos(database)
    print(" total num", pos)
    index_tokens = np.array(database[:pos, 0])
    np.save(output_path, index_tokens)
    print(" index path: {}".format(output_path))


'''
D: float[]
tokens: int[]
target_token:int
'''


def compute_KNN_prob_smart(D, tokens, k, target_token):
    last_value = D[k - 1]
    end_pos = 0
    D += 1e-6  # 避免全为0
    for i in range(k - 1, len(tokens)):
        if math.fabs(D[i] - last_value) < 1e-5:
            end_pos = i
    sum = np.sum(np.exp(-np.sqrt(D[:k])))
    target_num = np.where(tokens[:end_pos + 1] == target_token)[0]
    if len(target_num) > k:
        target_num = target_num[:k]
    tp = np.sum(np.exp(-np.sqrt(D[target_num])))

    return tp / sum


def compute_KNN_prob(D, tokens, T=1):
    prob_dict = {key: 0 for key in set(tokens)}
    D = D + 1e-6
    sum = 0
    for i in range(len(D)):
        token = tokens[i]
        dist = math.exp(-math.sqrt(D[i]) / T)  # e^-d
        sum += dist
        prob_dict[token] += dist
    for token in prob_dict.keys():
        prob_dict[token] = prob_dict[token] / sum
    return prob_dict


def compute_KNN_prob_only(D, tokens, k, target_token):
    D += 1e-6  # 避免全为0
    sum = np.sum(np.exp(-np.sqrt(D[:k])))
    target_num = np.where(tokens[:k] == target_token)[0]
    if len(target_num) > k:
        target_num = target_num[:k]
    tp = np.sum(np.exp(-np.sqrt(D[target_num])))
    return tp / sum


# D: batch * N
# ners: batch * N
# k: int
# target_token: batch *1
def compute_KNN_prob_only_batch(D, ners, k, target_token):
    D += 1e-6  # 避免全为0
    sum = np.sum(np.exp(-np.sqrt(D[:k])), axis=-1)
    target_lists = np.where(ners[:k] == target_token, True, False)  # batch * N
    tp = [np.sum(np.exp(-np.sqrt(D[i, :k][target_lists[i]]))) for i in target_lists.shape[0]] # batch *1
    return tp / sum



def find_last_pos(database):
    for index in range(len(database) - 1, 0, -1):
        if np.fabs(np.sum(database[index] - 0)) > 1e-6:
            return index + 1


def create_IVF(args):
    database_path = args.database_path
    shape = args.shape
    output_path = args.index_path
    dim = args.dim
    print("\n***************create IVF Index***************\n")
    quantizer = faiss.IndexFlatL2(dim)
    nlist = 100
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.nprobe = 10
    database_store = np.memmap(database_path, dtype=np.float32, mode="r",
                               shape=shape)
    pos = find_last_pos(database_store)
    print(" total num:", pos)
    print(" train")
    index.train(np.array(database_store[:pos, 1:]).astype("float32"))
    print(" add")
    index.add(np.array(database_store[:pos, 1:]).astype("float32"))
    print(" write")
    faiss.write_index(index, output_path)
    print("save to {}".format(output_path))



def create_IVF_and_tokens(args):
    create_IVF(args)
    create_index_tokens(args)


if __name__ == '__main__':
    # create_IVF("./database/clinc_train_200_per_mem.npy", shape = (3000000, 769), output_path="./database/clinc_index_200_IVF.index")
    create_index_tokens("./database/clinc_train_200_per_mem.npy", shape=(3000000, 769),
                        output_path="./database/clinc_tokens")
#
# index = faiss.read_index("./database/imdb_index_300_Ori.index")
# dur_t2 = test_index(index)
# print("IVF: ", dur_t1)
# print("ori: ", dur_t2)
#
