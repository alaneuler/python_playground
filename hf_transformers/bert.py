from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer("who are you?", return_tensors="pt")

model = AutoModel.from_pretrained(model_name)
output = model(**input, output_attentions=True)
print(output)
