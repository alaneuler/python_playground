from transformers import AutoModel, AutoTokenizer

model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

input_str = "这个天气可真是不错。"
print(tokenizer.tokenize(input_str))
input = tokenizer.encode(input_str, return_tensors="pt")
print(input)
print(tokenizer.decode(input[0][-1]))

output = model(input)
last_hidden_state = output.last_hidden_state
print(last_hidden_state)
