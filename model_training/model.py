from torch import nn


class SumModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)
