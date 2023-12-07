from multiprocessing import Process, set_start_method

import torch

input = torch.tensor([[1.0]], device="cuda")


def process(i, name):
    print(f"In {name} {i}...")
    input = torch.tensor([[1.0]], device=f"cuda:{i}")
    print(input)


if __name__ == "__main__":
    # set_start_method("spawn")
    set_start_method("fork")
    processes = [Process(target=process, args=(i, "test")) for i in range(3)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
