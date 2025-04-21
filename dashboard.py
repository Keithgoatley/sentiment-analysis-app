import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from rake_nltk import Rake

# --- Load models once ---
sentiment = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- Page layout ---
st.set_page_config(page_title="Sentiment Analytics Dashboard", layout="wide")
st.title("📊 Sentiment Analytics Dashboard")
st.markdown(
    """
    Upload or paste multiple reviews (one per line).  
    Get quantitative charts, top phrases, and a summary.
    """
)

# --- Input area ---
data_input = st.text_area("Paste reviews here:", height=200)
uploaded_file = st.file_uploader("Or upload a CSV (column named 'text')", type="csv")

# Read into DataFrame
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    texts = [t.strip() for t in data_input.split("\n") if t.strip()]
    df = pd.DataFrame({"text": texts})

if st.button("Run Analysis") and not df.empty:
    # 1) Sentiment inference
    results = sentiment(df["text"].tolist(), batch_size=16)
    df["label"] = [r["label"] for r in results]
    df["score"] = [r["score"] for r in results]

    # 2) Quantitative charts
    st.subheader("Quantitative Metrics")
    col1, col2 = st.columns(2)
    # Pie chart of labels
    counts = df["label"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%")
    ax1.set_title("Sentiment Distribution")
    col1.pyplot(fig1)

    # Histogram of confidence
    fig2, ax2 = plt.subplots()
    ax2.hist(df["score"], bins=10)
    ax2.set_title("Confidence Scores")
    ax2.set_xlabel("Score"); ax2.set_ylabel("Frequency")
    col2.pyplot(fig2)

    # 3) Keyword / aspect extraction via RAKE
    st.subheader("Top Key Phrases")
    r = Rake()
    all_text = " ".join(df["text"].tolist())
    r.extract_keywords_from_text(all_text)
    phrases = r.get_ranked_phrases()[:10]
    st.write(phrases)

    # 4) Executive summary (qualitative)
    st.subheader("Executive Summary")
    # summarize only the concatenated text
    summary = summarizer(all_text, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
    st.write(summary)

    # 5) Downloadable results
    st.subheader("Download Detailed Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="sentiment_analysis_results.csv", mime="text/csv")
