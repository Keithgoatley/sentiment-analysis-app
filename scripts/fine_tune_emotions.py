import os

import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

# Load dataset
train_df = pd.read_csv("data/goemotions_train.csv")
test_df = pd.read_csv("data/goemotions_test.csv")

# Binarize multi-label targets
mlb = MultiLabelBinarizer(classes=list(range(28)))
y_train = mlb.fit_transform(train_df["labels"].apply(eval))
y_test = mlb.transform(test_df["labels"].apply(eval))


# Define custom dataset class
class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# Create datasets
train_dataset = GoEmotionsDataset(train_df["text"].tolist(), y_train)
test_dataset = GoEmotionsDataset(test_df["text"].tolist(), y_test)

# Define model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=28, problem_type="multi_label_classification"
)

# Training arguments (simplified for older transformers version)
args = TrainingArguments(
    output_dir="models/emotion_final",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train
trainer.train()

# Save
os.makedirs("models/emotion_final", exist_ok=True)
model.save_pretrained("models/emotion_final")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained("models/emotion_final")

print("✔ Emotion model training complete and saved to models/emotion_final/")
