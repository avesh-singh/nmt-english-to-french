import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input_word):
        embedding = self.embedding(input_word)
        output, hidden = self.gru(embedding)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, input_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_word, hidden):
        embedding = self.relu(self.embedding(input_word)).view(1, 1, -1)
        # embedding = self.dropout(embedding)
        output, hidden = self.gru(embedding, hidden)
        output = self.output(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=0)
        self.linear = nn.Linear(hidden_size, hidden_size)
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        if self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = torch.randn(hidden_size, dtype=torch.float)

    def forward(self, hidden, encoder_states, device):
        seq_len = encoder_states.size(0)
        hidden = self.linear(hidden)
        attn_energies = torch.zeros(seq_len, device=device)
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden.view(-1), encoder_states[i], device)

        return self.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_state, device):
        try:
            # hidden @ encoder_state (no learnable params)
            if self.method == 'dot':
                return hidden.dot(encoder_state)
            # encoder_state @ W * hidden (learnable params)
            if self.method == 'general':
                energy = self.attn(hidden)
                return encoder_state.dot(energy)
            # v @ W * (hidden | encoder_state) (learnable params)
            if self.method == 'concat':
                energy = self.attn(torch.cat((hidden, encoder_state), -1))
                energy = nn.functional.tanh(energy)
                return self.v.to(device).dot(energy)
        except RuntimeError as r:
            print(hidden.size(), encoder_state.size())
            print(r)
            return torch.tensor(0)
