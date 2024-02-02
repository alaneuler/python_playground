from datasets import load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

model_path = "/data2/public_file/LLM_model/chinese-roberta-wwm-ext"
max_length = 10


def tokenize_map(examples):
    return tokenizer(
        examples["text"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )


tokenizer = BertTokenizer.from_pretrained(model_path)
data_set = load_dataset(
    "json",
    data_files={"train": "./data/train.json", "validation": "./data/val.json"},
)
data_set = data_set.map(tokenize_map, batched=True)

model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
)
model.train()

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=8,
    evaluation_strategy="epoch",
)
trainer = Trainer(
    model,
    args=training_args,
    train_dataset=data_set["train"],
    eval_dataset=data_set["validation"],
)

trainer.train()
trainer.save_model("./results/final_model")
