import math
import time
from data import EOS_index, SOS_index
import torch
import matplotlib.pyplot as plt

# plt.switch_backend('agg')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

MAX_TARGET_LENGTH = 10
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 150


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {round(s, 4)}s'


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return f'{as_minutes(s)} (- {as_minutes(rs)})'


def sentence_to_tensor(lang, s):
    idxs = [lang.word2index[word] for word in s.split()]
    idxs.append(EOS_index)
    return torch.tensor(idxs, device=device, dtype=torch.long).view(-1, 1)


def pair_to_tensor(pair, input_lang, output_lang):
    return sentence_to_tensor(input_lang, pair[0]), \
           sentence_to_tensor(output_lang, pair[1])


def get_decoder_input(output_tensor: torch.Tensor):
    t = torch.zeros_like(output_tensor)
    t[1:, 0] = output_tensor[:-1, 0].clone()
    return t


def show_plot(points):
    # plt.figure()
    # fig, ax = plt.subplots()
    plt.plot(points[1:])
    plt.show()
