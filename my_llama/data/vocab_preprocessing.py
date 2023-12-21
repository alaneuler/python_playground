import json

vocab = []
max_len = 0
with open("data/tang_poems_manual.json") as f:
    poems = json.load(f)

    vocab_set = set()
    for poem in poems:
        length = len(poem)
        if length > max_len:
            max_len = length
        for c in poem:
            vocab_set.add(c)
    vocab = list(vocab_set)

print(f"Vocab size: {len(vocab)}")
print(f"Poem max length: {max_len}")
with open("data/vocab.json", "w") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
