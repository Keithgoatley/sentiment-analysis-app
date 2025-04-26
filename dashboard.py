import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from transformers import pipeline
import yake

# --- Initialize once ---

# Sentiment analysis model
sentiment = pipeline(
    "sentiment-analysis",
    model="goatley/sentiment-final-model",
    tokenizer="goatley/sentiment-final-model"
)

# Emotion classification model
emotion = pipeline(
    "text-classification",
    model="models/emotion_final",
    tokenizer="models/emotion_final",
    return_all_scores=True
)

# Emotion ID to label mapping
emotion_id2label = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}

# --- Page layout ---
st.set_page_config(page_title="Sentiment & Emotion Analytics", layout="wide")
st.title("📊 Sentiment and Emotion Analytics Dashboard")
st.markdown("""
Upload or paste multiple reviews (one per line).  
Analyze sentiment and fine-grained emotional tones.
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
    # Sentiment inference
    sentiment_results = sentiment(df["text"].tolist(), batch_size=16)
    df["sentiment_label"] = [r["label"] for r in sentiment_results]
    df["sentiment_score"] = [r["score"] for r in sentiment_results]

    # Emotion inference
    emotion_results = emotion(df["text"].tolist(), batch_size=8)
    emotion_labels = [emotion_id2label[i] for i in range(28)]  # Use mapped labels
    for idx, emotions in enumerate(emotion_results):
        for emo_idx, emo in enumerate(emotions):
            df.loc[idx, emotion_labels[emo_idx]] = emo["score"]

    # Layout split
    st.subheader("Quantitative Metrics")
    col1, col2 = st.columns(2)

    # Sentiment distribution
    counts = df["sentiment_label"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%")
    ax1.set_title("Sentiment Distribution")
    col1.pyplot(fig1)

    # Top emotions
    emotion_scores = df[emotion_labels].mean().sort_values(ascending=False).head(5)
    fig2, ax2 = plt.subplots()
    ax2.bar(emotion_scores.index, emotion_scores.values)
    ax2.set_title("Top Emotions (average across reviews)")
    ax2.set_ylabel("Score")
    col2.pyplot(fig2)

    # Top keywords
    st.subheader("Top Key Phrases")
    all_text = " ".join(df["text"].tolist())
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=10)
    kw_pairs = kw_extractor.extract_keywords(all_text)
    keywords = [kw for kw, _ in kw_pairs]
    st.write(keywords)

    # Executive summary
    st.subheader("Executive Summary")
    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    scored = []
    for sent in sentences:
        score = sum(1 for kw in keywords if kw.lower() in sent.lower())
        scored.append((score, sent))
    top2 = [s for score, s in sorted(scored, key=lambda x: x[0], reverse=True) if score > 0][:2]
    if len(top2) < 2:
        top2 = sentences[:2]
    summary = " ".join(top2)
    st.write(summary)

    # Downloadable results
    st.subheader("Download Detailed Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="sentiment_emotion_analysis_results.csv",
        mime="text/csv"
    )
