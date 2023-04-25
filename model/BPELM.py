import os.path

import torch.nn as nn
import torch

from transformers import GPT2LMHeadModel

prefix = "/data1/gaoy/KNNLM-data"


class BPELM(nn.Module):
    def __init__(self, embedding_path, gpu):
        super().__init__()
        pre_embedding = torch.load(embedding_path,
                                   map_location= gpu).float()
        vocab_size, word_dim = pre_embedding.shape
        print("vocab_size:", vocab_size)
        self.word_embed = nn.Embedding(vocab_size, word_dim)
        self.word_embed.weight.data = pre_embedding
        self.word_embed.weight.requires_grad = True

        self.rnn = nn.LSTM(word_dim, 300, num_layers=1, bidirectional=False, batch_first=True)

        self.fc1 = nn.Linear(300, 100)
        self.fc = nn.Linear(100, vocab_size)
        # self.fc.weight.data = pre_embedding.clone()

    def forward(self, X):  # X (batch_size, seq_len)
        X = self.word_embed(X)
        hidden, _ = self.rnn(X)  # hidden (batch_size, seq_len, hidden_dim)
        hidden = self.fc1(hidden)  # hidden (batch_size, seq_len, 100)
        score = self.fc(hidden)
        return score

    def get_last_hidden(self, X):
        with torch.no_grad():
            X = self.word_embed(X)
            hidden, _ = self.rnn(X)  # hidden (batch_size, seq_len, hidden_dim)
            hidden = self.fc1(hidden)  # hidden (batch_size, seq_len, 100)
        return hidden


if __name__ == '__main__':
    path = os.path.join(prefix, "pretrained/gpt2")
    #model = GPT2LMHeadModel.from_pretrained(path)  # or any other checkpoint
    # word_embeddings = model.transformer.wte.weight.data  # Word Token Embeddings
    # word_embeddings = torch.cat((word_embeddings, torch.mean(word_embeddings, dim=0).reshape(1, -1)),
    #                             dim=0)  # last for [pad]
    # print(word_embeddings.shape)

    emb_path = "{}/data/embedding/{}_embedding.pt".format(prefix, "bpe")
    #torch.save(word_embeddings, emb_path)
    lm = BPELM(emb_path, -1)
    print(lm)
