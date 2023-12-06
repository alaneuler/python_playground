import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SumData
from model import SumModel
from utils import evaluation


def simple_training() -> nn.Module:
    hidden_size = 1000
    batch_size = 4

    model = SumModel(hidden_size)
    dataloader = DataLoader(SumData(), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), 1e-05)
    loss_fn = nn.MSELoss()
    for epoch in range(8):
        print(f"Epoch {epoch}")
        total_loss = 0
        model.train()
        for input, label in tqdm(dataloader):
            optimizer.zero_grad()
            pred = model(input)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(
            "Epoch average loss: %.3f"
            % (total_loss / len(dataloader) / batch_size)
        )
    return model


if __name__ == "__main__":
    start = time.time()
    model = simple_training()
    print(f"Training took {time.time() - start} seconds.")
    evaluation(model)

    model.eval()
    with torch.no_grad():
        test = torch.tensor([3, 2, 3], dtype=torch.float32)
        print(model(test))
        test = torch.tensor([3, 1, 3], dtype=torch.float32)
        print(model(test))
        test = torch.tensor([2, 11, 15], dtype=torch.float32)
        print(model(test))
