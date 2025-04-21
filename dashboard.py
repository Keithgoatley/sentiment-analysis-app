import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from transformers import pipeline
import yake

# --- Initialize once ---
sentiment = pipeline("sentiment-analysis")

# --- Page layout ---
st.set_page_config(page_title="Sentiment Analytics Dashboard", layout="wide")
st.title("📊 Sentiment Analytics Dashboard")
st.markdown("""
Upload or paste multiple reviews (one per line).  
Get quantitative charts, top phrases, and a detailed executive summary.
""")

# --- Inputs ---
data_input = st.text_area("Paste reviews here:", height=200)
uploaded_file = st.file_uploader("Or upload a CSV (column named 'text')", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    texts = [t.strip() for t in data_input.split("\n") if t.strip()]
    df = pd.DataFrame({"text": texts})

# --- Run Analysis ---
if st.button("Run Analysis") and not df.empty:
    # 1) Sentiment inference
    results = sentiment(df["text"].tolist(), batch_size=16)
    df["label"] = [r["label"] for r in results]
    df["score"] = [r["score"] for r in results]

    # 2) Quantitative charts
    st.subheader("Quantitative Metrics")
    col1, col2 = st.columns(2)

    counts = df["label"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%")
    ax1.set_title("Sentiment Distribution")
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(df["score"], bins=10)
    ax2.set_title("Confidence Scores")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Frequency")
    col2.pyplot(fig2)

    # 3) Keyword extraction via YAKE
    st.subheader("Top Key Phrases")
    all_text = " ".join(df["text"].tolist())
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=10)
    kw_pairs = kw_extractor.extract_keywords(all_text)
    keywords = [kw for kw, _ in kw_pairs]
    st.write(keywords)

    # 4) Executive summary via keyword-driven extractive method
    st.subheader("Executive Summary")
    # split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    # score each sentence on keyword overlap
    scored = []
    for sent in sentences:
        score = sum(1 for kw in keywords if kw.lower() in sent.lower())
        scored.append((score, sent))
    # pick top 2 sentences with score>0
    top2 = [s for score, s in sorted(scored, key=lambda x: x[0], reverse=True) if score > 0][:2]
    # fallback to first two sentences if needed
    if len(top2) < 2:
        top2 = sentences[:2]
    summary = " ".join(top2)
    st.write(summary)

    # 5) Downloadable results
    st.subheader("Download Detailed Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv"
    )
