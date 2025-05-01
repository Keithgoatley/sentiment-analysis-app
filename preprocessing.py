import pandas as pd
import re

def clean_text(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return ' '.join(text.split())        # normalize spaces

def load_data(path: str) -> pd.DataFrame:
    """
    Load raw CSV data from the given path.
    """
    return pd.read_csv(path)
