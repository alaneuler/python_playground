import torch
from data_source import TangPoemDataset, collote_fn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM

model_path = "./model"

config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_config(config)
print(
    "Number of parameters:",
    sum(p.numel() for p in model.parameters()),
)

dataset = TangPoemDataset("data/tang_poems_manual.json")
data_loader = DataLoader(dataset, batch_size=1, collate_fn=collote_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
model.train()

for epoch in range(20):
    for input_ids, label_ids in data_loader:
        optimizer.zero_grad()
        outputs = model(input_ids, labels=label_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, loss: {loss.data}")

model.save_pretrained(model_path)
