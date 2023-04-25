from torch.utils.data import Dataset
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import GPT2Tokenizer

max_length = 300
def get_file_dataset(file_path, tokenizers, is_with_eof = True):
    text_dataset = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            if is_with_eof:
                res = tokenizers(line.replace("\n", ''), return_tensors="pt", max_length=max_length, padding='max_length', truncation = True)
            else:
                line = "<|endoftext|>" + line
                res = tokenizers(line.replace("\n", ''), return_tensors="pt", max_length=max_length, padding='max_length', truncation = True)
            text_dataset.append(
                res.input_ids
            )
            line = f.readline()

    return text_dataset




class BPEDataset(Dataset):
    def __init__(self, fname, tokenizer, device, max_num = -1):
        super().__init__()
        EOS_IDX = 50256
        PAD_IDX = 50257
        self.device = device
        self.inputs = []
        self.targets = []
        self.lengths = []
        cnt = 0
        with open(fname, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip("\n")
                words = tokenizer(line, max_length=max_length, truncation = True).input_ids

                input = [EOS_IDX] + words
                input = input[:max_length]
                target = words[:max_length]
                length = len(input)
                input.extend([PAD_IDX] * (max_length - len(input)))
                target.extend([PAD_IDX] * (max_length - len(target)))
                self.inputs.append(input)
                self.targets.append(target)
                self.lengths.append(length)
                cnt += 1
                if max_num != -1 and cnt > max_num:
                    break

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx]), \
               self.lengths[idx]

    def __len__(self):
        return len(self.lengths)


class CLMDataset(Dataset):
    def __init__(self, fname, tokenizer, vocab):
        super().__init__()
        sents = []
        with open(fname, 'r') as f:
            line = f.readline()
            while line:
                sents.append(line.replace("\n", ''))
                line = f.readline()
        self.inputs = []
        self.targets = []
        self.lengths = []
        max_sequence_length = 300
        for sent in sents:
            words = tokenizer.tokenize(sent)
            input = ['<sos>'] + words
            input = input[:max_sequence_length]
            target = words[:max_sequence_length]  # without <sos>
            length = len(input)

            input.extend(['<pad>'] * (max_sequence_length - len(input)))
            target.extend(['<pad>'] * (max_sequence_length - len(target)))

            input = [vocab['w2i'].get(w, vocab['w2i']['<unk>']) for w in input]
            target = [vocab['w2i'].get(w, vocab['w2i']['<unk>']) for w in target]

            self.inputs.append(input)
            self.targets.append(target)
            self.lengths.append(length)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx]), \
               self.lengths[idx]

    def __len__(self):
        return len(self.lengths)