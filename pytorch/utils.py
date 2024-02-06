import torch
from torch import nn


def init3x2():
    linear = nn.Linear(2, 3)
    linear.weight = nn.Parameter(
        torch.tensor([[2.1915, 2.7888],
                      [1.9507, 2.0216],
                      [0.5517, 2.6375]])
    )
    return linear


def init1x3():
    linear = nn.Linear(3, 1)
    linear.weight = nn.Parameter(torch.tensor([[1.4572, 1.4801, 0.8898]]))
    return linear
