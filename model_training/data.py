import random

import torch
from torch.utils.data import Dataset


class SumData(Dataset):
    def __init__(self, size=5000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        "Data returned by this function should be tensors."

        i1 = random.randint(0, 20)
        i2 = random.randint(0, 20)
        i3 = random.randint(0, 20)
        return (
            torch.tensor([i1, i2, i3], dtype=torch.float32),
            torch.tensor([i1 + i2 + i3], dtype=torch.float32),
        )
