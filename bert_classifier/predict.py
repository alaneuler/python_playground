from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "/data2/public_file/LLM_model/chinese-roberta-wwm-ext"
)
model = BertForSequenceClassification.from_pretrained(
    "./results/final_model", num_labels=2
)

input = tokenizer("你好", return_tensors="pt")
output = model(**input)
cls = nn.functional.softmax(output.logits, dim=-1)
print(cls)
