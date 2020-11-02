from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from Lang import Lang


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    # de-couple end-of-sentence punctuations
    s = re.sub(r"([.!?])", r" \1", s)
    # remove irrelevant characters
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_languages(lang1, lang2, reverse=False):
    print("reading files...")

    lines = open("data/{}-{}.txt".format(lang1, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(l) for l in line.split('\t')] for line in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# filtering training examples
MAX_TRAINING_LENGTH = 10
eng_startswith = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
SOS_index = 0
EOS_index = 1


def filter_pair(p, eng_index):
    return len(p[0].split(' ')) < MAX_TRAINING_LENGTH \
           and len(p[1].split(' ')) < MAX_TRAINING_LENGTH \
           and p[eng_index].startswith(eng_startswith)


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_languages(lang1, lang2, reverse)
    eng_index = 1 if reverse else 0
    print("sentence pairs loaded: %d" % len(pairs))
    pairs = [pair for pair in pairs if filter_pair(pair, eng_index)]
    print("sentence pairs after filtering: %d" % len(pairs))
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Total words counted:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
