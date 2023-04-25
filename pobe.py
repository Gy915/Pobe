import numpy as np
import torch
from transformers import AutoConfig, AutoModelWithLMHead, GPT2LMHeadModel, AutoModelWithLMHead, GPT2Tokenizer, \
    AutoTokenizer, GPT2Config
from knn_faiss import get_faiss_and_token_index, compute_KNN_prob, compute_KNN_prob_smart, compute_KNN_prob_only
import torch.nn.functional as F
from util import *
import warnings
from tqdm import tqdm
from nltk import TweetTokenizer
from model.LM import LM
from model.BPELM import BPELM
from data import CLMDataset, BPEDataset
import copy
warnings.filterwarnings("ignore")
max_length = 1024


def get_sent_embedding(sent, tokenizers):
    return tokenizers(sent.replace("\n", ''), return_tensors="pt",
                      max_length=max_length, truncation=True).input_ids


def get_file_dataset(file_path, tokenizers, is_with_eof=True):
    text_dataset = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            if is_with_eof:
                res = tokenizers(line.replace("\n", ''), return_tensors="pt", max_length=max_length, truncation=True)
            else:
                line = "<|endoftext|>" + line
                res = tokenizers(line.replace("\n", ''), return_tensors="pt", max_length=max_length, truncation=True)
            text_dataset.append(
                res
            )
            line = f.readline()

    return text_dataset


class KNN_LM:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.faiss_index = None
        self.index_token = None
        self.model_type = args.model_type
        self.model_path = args.model_path
        # self.lambdas = args.lambdas
        self.device = args.gpu_model
        self.database_token_num = None
        self.back_model = None

    def set_up(self):
        model = None
        tokenizers = None
        if self.model_type == "gpt":
            config = GPT2Config.from_pretrained(self.model_path)
            model = GPT2LMHeadModel.from_pretrained(self.model_path, config=config)
            tokenizers = GPT2Tokenizer.from_pretrained(self.model_path)
            model.to(self.device)
            model.eval()
            model.config.output_hidden_states = True
            model.config.return_dict = True

            self.model = model
            self.tokenizer = tokenizers
        if self.model_type == "lstm":
            model = LM(vocab_size=args.vocab_num)
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            tokenizers = TweetTokenizer(preserve_case=False)
            model.to(self.device)
            model.eval()
            self.model = model
            self.tokenizer = tokenizers

        if self.model_type == "lstm_bpe":
            model = BPELM(embedding_path="/data1/gaoy/KNNLM-data/data/embedding/bpe_embedding.pt", gpu=self.device)
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            tokenizers = GPT2Tokenizer.from_pretrained("/data1/gaoy/KNNLM-data/pretrained/gpt2")
            model.to(self.device)
            model.eval()
            self.model = model
            self.tokenizer = tokenizers

        return model, tokenizers

    def handle_file_and_save(self):
        if self.args.type == "special_record":
            print("type =={}".format(self.args.type))
            for path in self.args.special_path:
                print("handle file:", path)
                if self.model_type == "lstm" or self.model_type == "lstm_bpe":
                    fine_name = path.split('/')[-2] + "_" + path.split('/')[-1].split(".txt")[0]
                else:
                    fine_name = path.split('/')[-1].split(".txt")[0]
                print(fine_name)
                res = self.record_probs_file(path)
                k_choose = args.Ks
                save_res_to_local(res, self.args, file_name=fine_name, k_choose=k_choose)

        if self.args.type == "knn" or self.args.type == "baseline":
            # handle id file:
            print("type =={}".format(self.args.type))
            print("handle id:", self.args.id_path)
            fine_name = args.id_path.split('/')[-1].split(".txt")[0]
            res_id = self.get_ppl_file(self.args.id_path)
            save_res_to_local(res_id, self.args, file_name=fine_name, res_shape=res_id.shape)

            print("handle ood:", self.args.ood_path)
            fine_name = args.ood_path.split('/')[-1].split(".txt")[0]
            res_ood = self.get_ppl_file(self.args.ood_path)
            save_res_to_local(res_ood, self.args, file_name=fine_name, res_shape=res_ood.shape)

        if self.args.type == "record":
            print("type =={}".format(self.args.type))
            print("handle id:", self.args.id_path)
            k_choose = args.Ks
            fine_name = args.id_path.split('/')[-1].split(".txt")[0]
            res_id = self.record_probs_file(self.args.id_path)
            save_res_to_local(res_id, self.args, file_name=fine_name, k_choose=k_choose)

            print("handle ood:", self.args.ood_path)
            fine_name = args.ood_path.split('/')[-1].split(".txt")[0]
            res_ood = self.record_probs_file(self.args.ood_path)
            save_res_to_local(res_ood, self.args, file_name=fine_name, k_choose=k_choose)

        if self.args.type == "log_ratio":
            print("type =={}".format(self.args.type))
            print("handle id:", self.args.id_path)
            fine_name = args.id_path.split('/')[-1].split(".txt")[0]
            res_id = self.get_ppl_file(self.args.id_path)
            save_res_to_local(res_id, self.args, file_name=fine_name)

            print("handle ood:", self.args.ood_path)
            fine_name = args.ood_path.split('/')[-1].split(".txt")[0]
            res_ood = self.get_ppl_file(self.args.ood_path)
            save_res_to_local(res_ood, self.args, file_name=fine_name)

        if self.args.type == "mixture":
            print("type =={}".format(self.args.type))
            print("handle id:", self.args.id_path)
            fine_name = args.id_path.split('/')[-1].split(".txt")[0]
            res_id = self.get_ppl_file(self.args.id_path)
            save_res_to_local(res_id, self.args, file_name=fine_name)

            print("handle ood:", self.args.ood_path)
            fine_name = args.ood_path.split('/')[-1].split(".txt")[0]
            res_ood = self.get_ppl_file(self.args.ood_path)
            save_res_to_local(res_ood, self.args, file_name=fine_name)

    def record_probs_file(self, file_path):

        if self.model is None:
            self.model, self.tokenizer = self.set_up()

        if self.faiss_index is None:
            self.faiss_index, self.index_token = get_faiss_and_token_index(self.args.tokens_path,
                                                                           self.args.index_path,
                                                                           is_shared=self.args.is_share_memory,
                                                                           gpu_list=self.args.gpu_knn_list)

        if self.model_type == "gpt":
            res_file = []
            dataset = get_file_dataset(file_path, self.tokenizer)
            file_n = len(dataset)
            for ix, input in tqdm(enumerate(dataset), total=file_n):
                encodings = input.input_ids  # 1 * seq
                res_sent = self.record_prob_sent(encodings)
                res_file.append(res_sent)

        if self.model_type == "lstm":
            res_file = {"lstm": [], "gpt_back": []}
            dataset = CLMDataset(file_path, self.tokenizer, vocab=json.load(open(args.vocab_path, 'r')))
            self.back_model_path = "/data1/gaoy/KNNLM-data/pretrained/gpt2"
            gpt_tokenizer = GPT2Tokenizer.from_pretrained(self.back_model_path)
            dataset_for_gpt = get_file_dataset(file_path, gpt_tokenizer, is_with_eof=False)

            for input, target, length in tqdm(dataset, desc="lstm"):
                encodings_lstm = input[: length].reshape(1, -1)  # 1 * seq
                res_lstm = self.record_prob_lstm_sent(encodings_lstm)
                res_file["lstm"].append(res_lstm)
            for input in tqdm(dataset_for_gpt, desc="gpt_back"):
                encodings = input.input_ids  # 1 * seq
                res_sent = self.record_prob_gpt_back_sent(encodings)
                res_file["gpt_back"].append(res_sent)
            assert len(res_file["gpt_back"]) == len(res_file["lstm"])

        if self.model_type == "lstm_bpe":
            res_file = {"lstm_bpe": [], "gpt_back": []}
            dataset = BPEDataset(file_path, self.tokenizer, self.device, max_num=-1)
            for input, target, length in tqdm(dataset, desc="lstm"):
                encodings_lstm = input[: length].reshape(1, -1)  # 1 * seq
                res_lstm = self.record_prob_lstm_sent(encodings_lstm)
                res_gpt = self.record_prob_gpt_back_sent(encodings_lstm)
                res_file["lstm_bpe"].append(res_lstm)
                res_file["gpt_back"].append(res_gpt)

        return res_file

    def get_pobe_res(self):
        print("========cal id scores=========")
        dataset = get_file_dataset(self.args.id_path, self.tokenizer)
        id_scores = self.get_pobe_scores(dataset, self.args.k)
        id_scores = [x for x in id_scores if np.isnan(x) == False and np.isinf(x) == False]
        print("======cal ood scores========")
        dataset = get_file_dataset(self.args.ood_path, self.tokenizer)
        ood_scores = self.get_pobe_scores(dataset, self.args.k)
        ood_scores = [x for x in ood_scores if np.isnan(x) == False and np.isinf(x) == False]
        labels = [0] * len(id_scores) + [1] * len(ood_scores)
        print(get_auc(labels, id_scores + ood_scores))

    def get_pobe_scores(self, dataset, k=1024):
        total_scores = []
        for ix, input in enumerate(tqdm(dataset)):
            encodings = input.input_ids  # 1 * seq
            p_lms, last_hidden_states = self.get_probs_lm_gpt(encodings)
            p_knns, _ = self.get_probs_knn_gpt_2(encodings, last_hidden_states, k)
            p_backs = self.get_probs_back_model_gpt(encodings)
            p_sent = np.zeros_like(p_lms)
            for pos, (p_lm, p_knn, p_back) in enumerate(zip(p_lms, p_knns, p_backs)):
                pobe_pos = (p_lm if p_lm > p_knn else p_knn) / p_back
                p_sent[pos] = pobe_pos
            total_scores.append(np.sum(-np.log(p_sent)))
        return total_scores

    def record_prob_gpt_back_sent(self, encodings):
        prob_back = self.get_probs_back_model_gpt(encodings)
        labels = encodings[0, 1:].numpy()  # seq-1
        labels_len = len(labels)
        res = []
        for pos in range(labels_len):
            dict_pos = {"token": labels[pos],
                        "prob_back": prob_back[pos]}
            res.append(dict_pos)
        return res

    '''
    
    Return: 
        res_sent:[dict_pos1, dict2_pos2, ...] 
            dict_posi:
                token: 0
                occur_num_in_database: np.int
                prob_lm：np.float
                prob_knn: np.array, [ 5, 20, 50, 100, 500]
                hit_num in:np.array, [5, 20, 50, 100, 500]
    '''

    def record_prob_sent(self, encodings):
        res = []
        if self.database_token_num == None:
            self.database_token_num = json.load(open(args.token_num_path, 'r'))
        if self.model_type == "gpt":
            probs_lm, last_hidden_states = self.get_probs_lm_gpt(encodings)  # seq-1, seq * dim
            probs_knn, ners = self.get_probs_knn_gpt(encodings,
                                                     last_hidden_states=last_hidden_states)  # seq-1 * ks
            prob_back = self.get_probs_back_model_gpt(encodings)
        if self.model_type == "lstm":
            probs_lm, last_hidden_states = self.get_probs_lm_lstm(encodings)  # seq-1, seq * dim
            probs_knn, ners = self.get_probs_knn_lstm(encodings,
                                                      last_hidden_states=last_hidden_states,
                                                      K=2048)  # prob_knn: seq-1 * ks;
            # ners: seq * ks
        labels = encodings[0, 1:].numpy()  # seq-1
        labels_len = len(labels)

        for pos in range(labels_len):
            # Ks: [5, 20, 50, 100, 200, 300, 500, 1024, 2048]
            dict_pos = {"token": labels[pos], "occur_num": self.database_token_num[str(labels[pos])],
                        "prob_lm": probs_lm[pos], "prob_knn": probs_knn[pos], "prob_back": prob_back[pos],
                        "hit_num": [len(np.where(ners[pos, :self.Ks[k_ix]] == labels[pos])[0]) for k_ix in
                                    range(len(self.Ks))]}
            res.append(dict_pos)
        return res

    '''
    Args:
        file_path: str
    Return:
        res: np.float, [file_n * lambdas_n * K_n]
            ppl for every sent in file
    '''

    def get_ppl_file(self, file_path):
        if self.model is None:
            self.model, self.tokenizer = self.set_up()

        if self.args.type == "knn":
            if self.faiss_index is None:
                self.faiss_index, self.index_token = get_faiss_and_token_index(self.args.tokens_path,
                                                                               self.args.index_path,
                                                                               is_shared=self.args.is_share_memory,
                                                                               gpu_list=self.args.gpu_knn_list)
            if self.model_type == "gpt":
                dataset = get_file_dataset(file_path, self.tokenizer)
                file_n = len(dataset)
                res = np.zeros((file_n, len(self.lambdas), len(self.Ks)), dtype=np.float)
                for ix, input in tqdm(enumerate(dataset), total=file_n):
                    encodings = input.input_ids  # 1 * seq
                    ppl_mix = self.get_ppl_mix_sent(encodings)
                    res[ix] = ppl_mix
            if self.model_type == "lstm":
                dataset = CLMDataset(file_path, self.tokenizer, vocab=json.load(open(args.vocab_path, 'r')))
                file_n = len(dataset)
                res = np.zeros((file_n, len(self.lambdas), len(self.Ks)), dtype=np.float)
                ix = 0
                for input, target, length in tqdm(dataset):
                    encodings = input[: length].reshape(1, -1)  # 1 * seq
                    ppl_mix = self.get_ppl_mix_sent(encodings)
                    res[ix] = ppl_mix
                    ix += 1

        if self.args.type == "baseline":
            if self.model_type == "lstm":
                dataset = CLMDataset(file_path, self.tokenizer, vocab=json.load(open(args.vocab_path, 'r')))
                file_n = len(dataset)
                res = np.zeros((file_n), dtype=np.float)
                ix = 0
                for input, target, length in tqdm(dataset):
                    encodings = input[: length].reshape(1, -1)  # 1 * seq
                    probs_lm, _ = self.get_probs_lm_lstm(encodings)
                    res[ix] = np.exp(np.mean(-np.log(probs_lm)))
                    ix += 1

        if self.args.type == "log_ratio":
            if self.model_type == "gpt":
                dataset = get_file_dataset(file_path, self.tokenizer)
                file_n = len(dataset)
                res = np.zeros((file_n), dtype=np.float)
                for ix, input in tqdm(enumerate(dataset), total=file_n):
                    encodings = input.input_ids  # 1 * seq
                    probs_lm, _ = self.get_probs_lm_gpt(encodings)
                    prob_back = self.get_probs_back_model_gpt(encodings)

                    res[ix] = -np.sum(np.log(probs_lm / prob_back))

        if self.args.type == "mixture":
            dataset = get_file_dataset(file_path, self.tokenizer)
            file_n = len(dataset)
            res = np.zeros((file_n, len(self.lambdas), len(self.Ks)), dtype=np.float)
            for ix, input in tqdm(enumerate(dataset), total=file_n):
                encodings = input.input_ids  # 1 * seq
                ppl_mix = self.get_ppl_mix_sent(encodings)
                res[ix] = ppl_mix
        return res

    '''
    args:
        encodings: 1 * seq
    return:
        mean_ppl: np, (lambdas_n * k_n)
    '''

    def get_ppl_mix_sent(self, encodings):
        if self.model_type == "gpt":
            probs_lm, last_hidden_states = self.get_probs_lm_gpt(encodings)  # seq-1
            probs_knn, _ = self.get_probs_knn_gpt(encodings,
                                                  last_hidden_states=last_hidden_states)  # seq-1 * ks

            if self.args.type == "mixture":
                prob_back = self.get_probs_back_model_gpt(encodings)

        if self.model_type == "lstm":
            probs_lm, last_hidden_states = self.get_probs_lm_lstm(encodings)  # seq-1
            probs_knn, _ = self.get_probs_knn_lstm(encodings,
                                                   last_hidden_states=last_hidden_states)  # seq-1 * ks

        seq_len = encodings.shape[1]
        res_sent = np.zeros((seq_len - 1, len(self.Ks), len(self.lambdas)))
        for l_ix, l in enumerate(self.lambdas):
            if self.args.type == "mixture":
                res_sent[..., l_ix] = (l * probs_lm[:, None] + (1 - l) * probs_knn[...]) / prob_back[:, None]
            if self.args.model_type == "knn":
                res_sent[..., l_ix] = l * probs_lm[:, None] + (1 - l) * probs_knn[...]
        res_sent = res_sent.transpose((0, 2, 1))
        if self.args.model_type == "knn":
            res = np.exp(np.mean(-np.log(res_sent), axis=0))
        if self.args.type == "mixture":
            res = np.sum(-np.log(res_sent), axis=0)
        return res

    '''
    Args:
        encodings: tensor, (1 * seq)
    Returns:
        prob_lms: np, (seq-1)
        last_hidden_states: np, (seq * hidden_dim)
    '''

    def get_probs_lm_lstm(self, encodings):
        encodings = encodings.to(self.device)
        outputs = self.model(encodings).detach().cpu()  # batch * seq * vocab_size
        labels = encodings[0, 1:].detach().cpu().numpy().astype(np.int)
        probs_lm = F.softmax(outputs[0, :-1], dim=-1).numpy()[np.arange(len(labels)), labels]
        last_hidden_states = self.model.get_last_hidden(encodings).squeeze().detach().cpu().numpy()
        return probs_lm, last_hidden_states

    '''
    Args:
        encodings: tensor, (1 * seq)
    Returns:
        prob_lms: np, (seq-1)
        last_hidden_states: np, (seq * hidden_dim)
    '''

    def get_probs_lm_gpt(self, encodings):
        outputs = self.model(encodings.to(self.device))
        logits_all = outputs["logits"].detach().cpu()  # 1 * seq * vocab_size
        labels = encodings[0, 1:].detach().cpu().numpy().astype(np.int)
        probs_lm = F.softmax(logits_all[0, :-1], dim=-1).numpy()[np.arange(len(labels)), labels]
        try:
            assert probs_lm.shape[0] == (encodings.shape[1] - 1)
        except Exception as e:
            print("logits_all:", logits_all.shape)
            print("labels:", labels.shape)
            print("probs_lm", probs_lm.shape)
        last_hidden_states = outputs["hidden_states"][-1][0].detach().cpu().numpy()
        return probs_lm, last_hidden_states

    '''
    Args:
        encodings: tensor, (1 * seq)
        last_hidden_state: tensor, (seq * hidden_dim)
    Returns:
        probs_knn: np, (seq-1, ks)
    '''

    def get_probs_knn_gpt(self, encodings, last_hidden_states=None, K=None):
        if last_hidden_states is None:
            encodings = encodings.to(self.device)
            outputs = self.model(encodings)
            last_hidden_states = outputs["hidden_states"][-1][0].detach().cpu().numpy()
        if self.faiss_index is None:
            self.faiss_index, self.index_token = get_faiss_and_token_index(self.args.tokens_path, self.args.index_path,
                                                                           is_shared=self.args.is_share_memory,
                                                                           gpu_list=self.args.gpu_knn_list)
        probs_knn = np.zeros((encodings.shape[1] - 1, len(self.Ks)))
        max_k = self.Ks[-1] if K is None else K
        D_all, I_all = self.faiss_index.search(last_hidden_states, max_k)  # seq * K
        ners = self.index_token[I_all].astype(np.int)
        labels = encodings[0, 1:].detach().cpu().numpy().astype(np.int)
        for ix, label in enumerate(labels):
            for kx, k in enumerate(self.Ks):
                if k <= max_k:
                    probs_knn[ix, kx] = compute_KNN_prob_only(D_all[ix, :k], ners[ix, :k], k, label)
        return probs_knn, ners

    # return: knn_p: (seq - 1)
    def get_probs_knn_gpt_2(self, encodings, last_hidden_states=None, K=None):
        if last_hidden_states is None:
            encodings = encodings.to(self.device)
            outputs = self.model(encodings)
            last_hidden_states = outputs["hidden_states"][-1][0].detach().cpu().numpy()
        if self.faiss_index is None:
            self.faiss_index, self.index_token = get_faiss_and_token_index(self.args.tokens_path, self.args.index_path,
                                                                           is_shared=self.args.is_share_memory,
                                                                           gpu_list=self.args.gpu_knn_list)
        probs_knn = np.zeros(shape=(encodings.shape[1] - 1))
        D_all, I_all = self.faiss_index.search(last_hidden_states, K)  # seq * K
        ners = self.index_token[I_all].astype(np.int)
        labels = encodings[0, 1:].detach().cpu().numpy().astype(np.int)
        for ix, label in enumerate(labels):
            probs_knn[ix] = compute_KNN_prob_only(D_all[ix], ners[ix], K, label)
        return probs_knn, ners

    def get_probs_back_model_gpt(self, encodings):
        if self.back_model is None:
            self.back_model_path = self.args.pretrained_path
            self.back_model = GPT2LMHeadModel.from_pretrained(self.back_model_path)
            self.back_model.to(self.device)

        outputs = self.back_model(encodings.to(self.device))
        logits_all = outputs["logits"].detach().cpu()  # 1 * seq * vocab_size
        labels = encodings[0, 1:].detach().cpu().numpy().astype(np.int)
        probs_lm = F.softmax(logits_all[0, :-1], dim=-1).numpy()[np.arange(len(labels)), labels]
        return probs_lm

    def get_probs_back_model_lstm(self, encodings):
        if self.back_model is None:

            self.back_model_path = "pretrained/gpt2"
            if self.args.data_root != "None":
                self.back_model_path = os.path.join(args.data_root, self.back_model_path)
            self.back_model = GPT2LMHeadModel.from_pretrained(self.back_model_path)
            self.back_model.to(self.device)
        outputs = self.back_model(encodings.to(self.device))
        logits_all = outputs["logits"].detach().cpu()  # 1 * seq * vocab_size
        labels = encodings[0, 1:].detach().cpu().numpy().astype(np.int)
        probs_lm = F.softmax(logits_all[0, :-1], dim=-1).numpy()[np.arange(len(labels)), labels]
        return probs_lm


if __name__ == '__main__':
    args = load_args("config/pobe.yaml")

    pprint_config(args)
    knn_lm = KNN_LM(args)
    knn_lm.set_up()
    knn_lm.get_pobe_res()
