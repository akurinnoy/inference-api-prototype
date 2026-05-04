import json


class Tokenizer:
    PAD = 0
    UNK = 1

    def __init__(self):
        self.word2idx = {"<PAD>": self.PAD, "<UNK>": self.UNK}
        self.idx2word = {self.PAD: "<PAD>", self.UNK: "<UNK>"}

    def build_vocab(self, texts):
        idx = len(self.word2idx)
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    @property
    def vocab_size(self):
        return len(self.word2idx)

    def encode(self, text):
        return [self.word2idx.get(w, self.UNK) for w in text.lower().split()]

    def decode(self, indices):
        return [self.idx2word.get(i, "<UNK>") for i in indices]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def load(self, path):
        with open(path) as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
