import os
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

def main():
    # 1) Load the IMDB CSV subsets
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "data/imdb_train.csv",
            "test":  "data/imdb_test.csv"
        }
    )

    # 2) Initialize DistilBERT and its tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # 3) Tokenization helper
    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # 4) Apply tokenization to both train and test sets
    encoded = dataset.map(preprocess, batched=True)

    # 5) Load accuracy metric via 'evaluate'
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # 6) Define training arguments (compatible with older transformers)
    training_args = TrainingArguments(
        output_dir                 = "models/sentiment",
        num_train_epochs           = 3,
        per_device_train_batch_size= 16,
        per_device_eval_batch_size = 16,
        logging_dir                = "logs",
        logging_steps              = 50,
        save_steps                 = 500
    )

    # 7) Initialize the Trainer
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = encoded["train"],
        eval_dataset    = encoded["test"],
        tokenizer       = tokenizer,
        compute_metrics = compute_metrics
    )

    # 8) Run fine‑tuning
    trainer.train()
    print("✔ Fine‑tuning complete.")

    # 9) Evaluate on the test set
    results = trainer.evaluate()
    print("Evaluation results:", results)

    # 10) Save the final model
    trainer.save_model("models/sentiment_final")
    print("✔ Model saved to models/sentiment_final/")

if __name__ == "__main__":
    main()
