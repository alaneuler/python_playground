import torch
from transformers import AutoModelForCausalLM

from tokenizer import decode, encode

model_path = "./model"

model = AutoModelForCausalLM.from_pretrained(model_path)

while True:
    query = input("Enter start: ")
    if query == "":
        continue
    if query == "q":
        break

    input_ids = torch.tensor([encode(query)])
    generated_ids = model.generate(
        input_ids,
        max_length=100,
    )
    print(decode(generated_ids[0].tolist()))
