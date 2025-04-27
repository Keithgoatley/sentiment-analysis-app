import os

import pandas as pd
from datasets import load_dataset

# Create a data/ directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load GoEmotions dataset
dataset = load_dataset("go_emotions")

# Use only a smaller subset for faster fine-tuning
train = dataset["train"].shuffle(seed=42).select(range(5000))
test = dataset["test"].shuffle(seed=42).select(range(1000))

# Save small subset as CSV
train_df = pd.DataFrame({"text": train["text"], "labels": train["labels"]})
test_df = pd.DataFrame({"text": test["text"], "labels": test["labels"]})

train_df.to_csv("data/goemotions_train.csv", index=False)
test_df.to_csv("data/goemotions_test.csv", index=False)

print("Saved data/goemotions_train.csv and data/goemotions_test.csv")
