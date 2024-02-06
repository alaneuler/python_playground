import torch
from torch import nn
from utils import init1x3, init3x2

network = nn.Sequential(
    init3x2(),
    nn.ReLU(),
    init1x3(),
    nn.ReLU(),
)

# 2*3+3 + 3*1+1 = 13
parameter_num = 0
for p in network.parameters():
    parameter_num += p.numel()
    print(p)
print(f"Parameter num: {parameter_num}\n")

input = torch.tensor([[1.0, 2.0]])
output = network(input)
print(f"Output: {output}")

criterion = nn.MSELoss()
loss = criterion(output, torch.tensor([[1.0]]))
loss.backward()
print(loss)
