import json

import torch
from torch.utils.data import Dataset

from tokenizer import encode


class TangPoemDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        with open(path) as f:
            self.poems = json.load(f)

    def __len__(self):
        return len(self.poems)

    def __getitem__(self, idx):
        return self.poems[idx]


def collote_fn(batch):
    input_ids = []
    label_ids = []
    for poem in batch:
        ids = encode(poem)
        input_ids.append(ids)
        label_ids.append(ids)

    return torch.tensor(input_ids), torch.tensor(label_ids)
