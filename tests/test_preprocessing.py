import pandas as pd
import re
import pytest
from preprocessing import clean_text, load_data

def test_clean_text_basic():
    assert clean_text("Hello, WORLD!!!") == "hello world"

def test_clean_text_whitespace():
    assert clean_text("  Foo   Bar  ") == "foo bar"

def test_load_data_and_drop(tmp_path):
    # create a sample CSV with a missing review
    data = tmp_path / "sample.csv"
    data.write_text("review_text,label\nTest review,positive\n,neutral")
    df = load_data(str(data))
    # load_data should read both rows, but later pipelines drop nulls
    assert list(df.columns) == ["review_text", "label"]
    assert df.shape[0] == 2  # raw load; nulls still there
