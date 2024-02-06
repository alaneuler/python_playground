import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data import SumData
from model import SumModel
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import evaluation


def setup(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def split_dataloader(rank: int, world_size: int, batch_size: int):
    dataset = SumData()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def wrap_model(device, hidden_size):
    model = SumModel(hidden_size).to(device)
    model = DDP(model, device_ids=[device])
    return model


def cleanup():
    dist.destroy_process_group()


def train_main(rank, world_size, hidden_size):
    # Set the batch size to 1 so global batch size is 4
    batch_size = 1

    setup(rank, world_size)
    dataloader = split_dataloader(rank, world_size, batch_size)
    model = wrap_model(rank, hidden_size)

    optimizer = Adam(model.parameters(), 1e-05)
    loss_fn = nn.MSELoss()

    print(f"Train on device {rank}")
    for epoch in range(8):
        dataloader.sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        for input, label in dataloader:
            input, label = input.to(rank), label.to(rank)

            optimizer.zero_grad()
            pred = model(input)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(
            "Epoch %d on %d average loss: %.3f"
            % (
                epoch,
                rank,
                (total_loss / len(dataloader) / batch_size),
            )
        )
    cleanup()

    if rank == 0:
        torch.save(model.module.state_dict(), "output/model_ddp.pt")


if __name__ == "__main__":
    model_path = "output/model_ddp.pt"
    hidden_size = 1000
    if not os.path.exists(model_path):
        start = time.time()
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        mp.spawn(
            train_main,
            nprocs=world_size,
            args=(world_size, hidden_size),
        )
        print(f"Training took: {time.time() - start} seconds.")

    model = SumModel(hidden_size)
    model.load_state_dict(torch.load(model_path))
    evaluation(model)

    model.eval()
    with torch.no_grad():
        test = torch.tensor([3, 2, 3], dtype=torch.float32)
        print(model(test))
        test = torch.tensor([3, 1, 3], dtype=torch.float32)
        print(model(test))
        test = torch.tensor([2, 11, 15], dtype=torch.float32)
        print(model(test))

    os.remove(model_path)
