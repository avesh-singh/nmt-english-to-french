import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input_word, hidden):
        embedding = self.embedding(input_word).view(1, 1, -1)
        output, hidden = self.gru(embedding, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, input_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_word, hidden):
        embedding = self.relu(self.embedding(input_word)).view(1, 1, -1)
        output, hidden = self.gru(embedding, hidden)
        output = self.output(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
