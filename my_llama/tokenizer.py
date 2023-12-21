import json

with open("data/vocab_final.json") as f:
    vocab = json.load(f)

vocab_itos = {idx: ch for idx, ch in enumerate(vocab)}
vocab_stoi = {ch: idx for idx, ch in enumerate(vocab)}


def encode(s):
    return [vocab_stoi[ch] for ch in s]


def decode(ids):
    return "".join([vocab_itos[i] for i in ids])
