import torch
from torchcrf import CRF

num_tags = 5
model = CRF(num_tags)

seq_length = 3
batch_size = 1
emissions = torch.randn(
    seq_length, batch_size, num_tags
)
tags = torch.tensor([
    [0], [2], [3],
    # [1, 4, 1]
])
print(model(emissions, tags))

print(model.decode(emissions))
