import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead, GPT2LMHeadModel, AutoModelWithLMHead
import numpy as np
from tqdm import tqdm

model_path = "./ckpts/gpt2-clinc"
file_path = "./dataset/clinc/clinc_train_gpt.txt"


def get_file_dataset(tokenizers, file_name):
    text_dataset = []
    cnt = 0
    with open(file_name, 'r') as f:
        line = f.readline()
        while line:
            #print(cnt)
            text_dataset.append(tokenizers(line.replace("\n", ''), return_tensors="pt", max_length=1024))
            line = f.readline()
            cnt += 1
    return text_dataset


# input -> GPT -> output
# embedding -> key
# tokenID -> value
# database = [[embedding + tokenID], [], []]
def gather_keys(args):
    print("***********gather keys*************")
    model_path = args.model_path
    device = args.device
    model = AutoModelWithLMHead.from_pretrained(model_path)
    model.to(device)
    model.eval()
    model.config.return_dict = True
    model.config.output_hidden_states = True
    dim = args.dim
    output_path = args.database_path
    tokenizers = AutoTokenizer.from_pretrained(model_path)
    file_path = args.train_path
    print(" loading data from {}".format(file_path))
    dataset = get_file_dataset(file_name=file_path, tokenizers=tokenizers)
    sent_num = len(dataset)
    per_sent_word = args.per_sent
    num_totoal_limt = sent_num * per_sent_word
    print("   database limit length: ", num_totoal_limt)
    print("   output_path:", output_path)
    database_store = np.memmap(output_path, dtype=np.float32, mode="w+", shape=(num_totoal_limt, dim + 1))
    dataset = tqdm(dataset)
    cnt = 0
    for i in dataset:
        input_ids = i["input_ids"].to(device)
        sent_len = input_ids.shape[1]
        outputs = model(input_ids)
        last_hidden_state = outputs["hidden_states"][-1][0]  # seq * dim
        for j in range(0, sent_len - 1):
            if j >= per_sent_word:
                break
            token_id = input_ids[0][j + 1]  # val
            sent_embeding = last_hidden_state[j]
            data = np.zeros((dim + 1), dtype=np.float)
            data[0] = token_id.item()
            data[1:] = sent_embeding.detach().cpu().numpy()
            database_store[cnt][:] = data
            cnt += 1

    database_store.flush()
    print("    database shape: ", num_totoal_limt, " * ", dim + 1)
    print("************ database save to {} **************".format(output_path))
    return (num_totoal_limt, dim + 1)


if __name__ == '__main__':
    tokenizers = AutoTokenizer.from_pretrained(model_path)
    dataset = get_file_dataset(file_name=file_path, tokenizers=tokenizers)
    model = AutoModelWithLMHead.from_pretrained(model_path)
    device = torch.device("cuda:0")
    gather_keys(model, dataset, device, output_path="./database/clinc_train_200_per")
