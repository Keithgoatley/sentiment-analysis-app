import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch
import shap
import streamlit.components.v1 as components
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import yake
from semantic_search import SemanticSearch
from bertopic import BERTopic

# --- Initialize models once ---

# Sentiment analysis pipeline (local fine-tuned, CPU)
sentiment = pipeline(
    "sentiment-analysis",
    model="models/sentiment_final",
    tokenizer="models/sentiment_final",
    device=-1
)

# Emotion classification pipeline (local fine-tuned, CPU)
emotion = pipeline(
    "text-classification",
    model="models/emotion_final",
    tokenizer="models/emotion_final",
    return_all_scores=True,
    device=-1
)

# Load raw model + tokenizer for SHAP
sent_model = AutoModelForSequenceClassification.from_pretrained("models/sentiment_final")
sent_tokenizer = AutoTokenizer.from_pretrained("models/sentiment_final")

def predict_sentiment(texts):
    """Return raw logits for SHAP."""
    enc = sent_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = sent_model(**enc).logits
    return logits.cpu().numpy()

# Emotion ID → human label
emotion_id2label = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}

# Initialize semantic search
if 'semantic_search' not in st.session_state:
    st.session_state.semantic_search = SemanticSearch()

# Emotion labels in session
if 'emotion_labels' not in st.session_state:
    st.session_state.emotion_labels = [emotion_id2label[i] for i in range(28)]

# Track if analysis has been run
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

# --- Page layout ---
st.set_page_config(page_title="Sentiment & Emotion Analytics", layout="wide")
st.title("📊 Sentiment & Emotion Analytics Dashboard")
st.markdown("""
Upload or paste multiple reviews (one per line).  
Analyze sentiment, fine-grained emotional tones, semantic similarity, topics, and explanations.
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
    # 1) Sentiment
    sentiment_results = sentiment(df["text"].tolist(), batch_size=16)
    df["sentiment_label"] = [r["label"] for r in sentiment_results]
    df["sentiment_score"] = [r["score"] for r in sentiment_results]

    # 2) Emotion
    emotion_results = emotion(df["text"].tolist(), batch_size=8)
    for i, ems in enumerate(emotion_results):
        for j, e in enumerate(ems):
            df.loc[i, st.session_state.emotion_labels[j]] = e["score"]

    # 3) Keywords (YAKE)
    all_text = " ".join(df["text"])
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=10)
    keywords = [kw for kw, _ in kw_extractor.extract_keywords(all_text)]

    # 4) Executive summary
    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    scored = [(sum(1 for kw in keywords if kw.lower() in s.lower()), s) for s in sentences]
    top2 = [s for sc, s in sorted(scored, key=lambda x: x[0], reverse=True) if sc>0][:2]
    if len(top2)<2: top2 = sentences[:2]
    summary = " ".join(top2)

    # 5) Semantic search index
    st.session_state.semantic_search.build_index(df["text"].tolist())

    # 6) Topic modeling (BERTopic)
    topic_model = BERTopic(language="english", calculate_probabilities=True)
    topics, _ = topic_model.fit_transform(df["text"].tolist())
    df["topic"] = topics
    st.session_state.topics = topics
    st.session_state.topic_model = topic_model

    # 7) SHAP explainer (background = first 100 texts)
    st.session_state.shap_explainer = shap.Explainer(predict_sentiment, df["text"].tolist()[:100])

    # 8) Save in session
    st.session_state.df       = df
    st.session_state.keywords = keywords
    st.session_state.summary  = summary
    st.session_state.analysis_run = True

# --- Display Results ---
if st.session_state.analysis_run:
    df            = st.session_state.df
    keywords      = st.session_state.keywords
    summary       = st.session_state.summary
    emotion_labels= st.session_state.emotion_labels
    topics        = st.session_state.topics

    st.subheader("Quantitative Metrics")
    col1, col2, col3 = st.columns(3)

    # Sentiment pie
    counts = df["sentiment_label"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%")
    ax1.set_title("Sentiment Distribution")
    col1.pyplot(fig1)

    # Emotions bar
    emo_scores = df[emotion_labels].mean().sort_values(ascending=False).head(5)
    fig2, ax2 = plt.subplots()
    ax2.bar(emo_scores.index, emo_scores.values)
    ax2.set_title("Top Emotions")
    ax2.set_ylabel("Score")
    col2.pyplot(fig2)

    # Topics bar
    top_topic_counts = pd.Series(topics).value_counts().head(5)
    fig3, ax3 = plt.subplots()
    ax3.bar(top_topic_counts.index.astype(str), top_topic_counts.values)
    ax3.set_title("Top Topics (BERTopic)")
    ax3.set_ylabel("Count")
    col3.pyplot(fig3)

    # Keywords
    st.subheader("Top Key Phrases")
    st.write(keywords)

    # Summary
    st.subheader("Executive Summary")
    st.write(summary)

    # Download CSV
    st.subheader("Download Detailed Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")

    # Explainable AI
    st.subheader("🧪 Explainability (SHAP)")
    for idx, text in enumerate(df["text"]):
        st.markdown(f"**Review #{idx+1}:** {text}")
        if st.button(f"Explain sentiment #{idx+1}", key=f"shap_{idx}"):
            sv = st.session_state.shap_explainer([text])
            html = shap.plots.text(sv[0], display=False)
            components.html(html.data, height=200)

    # Semantic Search
    st.subheader("🔍 Semantic Search")
    q = st.text_input("Enter query:", key="sem_q")
    if st.button("Search Similar Reviews"):
        if q.strip():
            res = st.session_state.semantic_search.query(q, top_k=5)
            st.write("#### Similar Reviews:")
            for t, d in res:
                st.markdown(f"- {t} *(score {d:.3f})*")
        else:
            st.write("Please enter a query.")
