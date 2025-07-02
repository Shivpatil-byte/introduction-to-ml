# sentiment_pipeline.py

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np

# Step i: Load IMDb dataset
dataset = load_dataset("imdb")

# Step ii: Preprocess dataset with tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Step iii: Load model for binary classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Step iv: Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    save_strategy="epoch"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(2000)),  # small subset for demo
    eval_dataset=tokenized_datasets["test"].select(range(500)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Step v: Save model and tokenizer
model.save_pretrained("sentiment-model")
tokenizer.save_pretrained("sentiment-model")

# Load and test inference
text = "The movie was absolutely wonderful!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits).item()
print("Predicted Sentiment:", "positive" if prediction == 1 else "negative")
