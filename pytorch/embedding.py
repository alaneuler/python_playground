import torch

data = [
    ("cat", "meow"),
    ("dog", "bark"),
    ("bird", "fly"),
    ("cat", "drink"),
    ("dog", "eat"),
]

word2idx = {word: i for i, word in enumerate(set(sum(data, ())))}
vocab_size = len(word2idx)
embedding_dim = 5

model = torch.nn.Embedding(vocab_size, embedding_dim)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for target, context in data:
        input_ids = torch.tensor([[word2idx[context]]])
        labels = torch.tensor([[word2idx[target]]])

        optimizer.zero_grad()
        output = model(input_ids)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

print(model.weight.data)
