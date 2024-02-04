import torch
from torch.nn import CrossEntropyLoss

criterion = CrossEntropyLoss()
loss = criterion(torch.tensor([0.1, 0.9, 0.8]), torch.tensor(1))
print(loss)

loss = criterion(
    torch.tensor([[0.2, 1.1, 0.1], [0.1, 0.9, 0.8]]), torch.tensor([2, 1])
)
print(loss)
