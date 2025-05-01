import pandas as pd
import re


def clean_text(text: str) -> str:
    """
    Normalize and clean raw text.

    Steps:
    1. Convert to lowercase.
    2. Remove punctuation.
    3. Collapse multiple whitespace into a single space.

    Args:
        text: Raw input string.

    Returns:
        A cleaned, lowercase string with no punctuation and normalized spaces.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw CSV data from a file path.

    Args:
        path: Filesystem path to a CSV file containing review data.

    Returns:
        A pandas DataFrame loaded from the CSV.
    """
    return pd.read_csv(path)
