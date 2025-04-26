from datasets import load_dataset

# 1) Load the full IMDB dataset
dataset = load_dataset("imdb")

# 2) Shuffle and select a smaller subset for faster fine-tuning
small_train = dataset["train"].shuffle(seed=42).select(range(5000))
small_test  = dataset["test"].shuffle(seed=42).select(range(1000))

# 3) Save these subsets as CSV files in the data/ folder
small_train.to_csv("data/imdb_train.csv", index=False)
small_test.to_csv("data/imdb_test.csv",  index=False)

print("Saved data/imdb_train.csv and data/imdb_test.csv")
