import torch.nn as nn
import torch

class LM(nn.Module):
    def __init__(self, vocab_size, word_dim=100):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, word_dim)
        self.rnn = nn.LSTM(word_dim, 500, num_layers=1, bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(500, word_dim)
        self.fc = nn.Linear(word_dim, vocab_size)
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
