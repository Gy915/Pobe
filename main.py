from fintune_gpt import fine_tune
import argparse
from KNN_gather import gather_keys
from transformers import AutoModelWithLMHead
from knn_faiss import create_IVF_and_tokens
import warnings

warnings.filterwarnings("ignore")


def main():
    dataset = "imdb"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--train_path", type=str, default="./dataset/{}/{}_train_gpt.txt".format(dataset, dataset))
    parser.add_argument("--eval_path", type=str, default="./dataset/{}/{}_valid_gpt.txt".format(dataset, dataset))
    parser.add_argument("--per_sent", type=int, default=300)
    parser.add_argument("--dim", type=int, default=768)
    # parser.add_argument("--shape", type=tuple, default=(7500000, 769))
    parser.add_argument("--device", type=str, default="cuda:0")

    # if machine == "2080ti1":
    #     parser.add_argument("--pretrained_path", type=str, default="./pretrained/gpt2")
    #     parser.add_argument("--tokens_path", type=str, default="./database/{}_tokens_final.npy".format(dataset))
    #     parser.add_argument("--model_path", type=str, default="./ckpts/{}_final".format(dataset))
    #     parser.add_argument("--database_path", type=str,
    #                         default="./database/{}_train_{}_per_mem_final.npy".format(dataset,
    #                                                                                   300))
    #     parser.add_argument("--index_path", type=str, default="./database/{}_index_{}_IVF_final.index".format(dataset,
    #                                                                                                           300 ))


    parser.add_argument("--pretrained_path", type=str, default="./pretrained/gpt2")
    parser.add_argument("--model_prefix", type=str,
                        default="./ckpts/")
    parser.add_argument("--database_prefix", type=str,
                        default="/data1/gaoy/KNNLM-data/database/")
    args = parser.parse_args()
    args.model_path = args.model_prefix + f"{args.dataset}_final"
    args.database_path = args.database_prefix + f"{args.dataset}_{args.per_sent}_mem_final.npy"
    args.index_path = args.database_prefix + f"{args.dataset}_{args.per_sent}_IVF_final.index"
    args.tokens_path = args.database_prefix + f"{args.dataset}_{args.per_sent}_tokens_final.npy"

    print(args)
    fine_tune(args)
    shape = gather_keys(args)
    args.shape = shape
    create_IVF_and_tokens(args)


if __name__ == '__main__':
    main()
