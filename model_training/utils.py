from torch import nn
from torch.utils.data import DataLoader

from data import SumData


def evaluation(model: nn.Module):
    batch_size = 4

    model.train()
    dataloader = DataLoader(SumData(100), batch_size=batch_size)
    loss_fn = nn.MSELoss()
    total_loss = 0
    for input, label in dataloader:
        pred = model(input)
        loss = loss_fn(pred, label)
        total_loss += loss.item()
    print(
        "Evaluation average loss: %.5f"
        % (total_loss / len(dataloader) / batch_size)
    )
