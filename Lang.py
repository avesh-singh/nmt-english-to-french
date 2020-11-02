class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>": 0, "<EOS>": 1}
        self.index2word = {0: "<SOS>", 1: "<EOS>"}
        self.word2count = {}
        self.n_words = 2

    def add_sentence(self, sentence):
        _ = [self.add_word(word) for word in sentence.split()]

    def add_word(self, word):
        if word not in self.word2index.keys():
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        self.word2count[word] += 1
