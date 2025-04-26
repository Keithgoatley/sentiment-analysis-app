# 📊 Sentiment and Emotion Analytics Dashboard

An advanced, production-grade Natural Language Processing (NLP) dashboard that analyzes text reviews to extract:
- Overall **sentiment** (positive or negative)
- Fine-grained **emotions** (joy, sadness, anger, etc.)
- **Key phrases** and topics
- **Executive summaries** based on semantic importance

Built as part of my MSc Artificial Intelligence application portfolio to demonstrate applied skills in deep learning, NLP, data engineering, and full-stack development.

---

## 🛠 Features

- **Sentiment Analysis**: Fine-tuned DistilBERT model classifying positive/negative sentiment
- **Emotion Classification**: Multi-label emotion detection across 27 emotional categories + neutral
- **Keyword Extraction**: Automated top phrases via YAKE
- **Executive Summary**: Extractive summarization based on semantic relevance
- **Modern UI**: Built with Streamlit for a clean, responsive layout
- **Model Deployment**: Hugging Face Hub integration
- **Downloadable Results**: CSV export for batch review analysis
- **Fully reproducible code** in GitHub and Codespaces

---

## 🧠 How It Works

1. User pastes text reviews or uploads a CSV.
2. Two models are run in parallel:
   - Fine-tuned **sentiment model** (positive/negative classification)
   - Fine-tuned **emotion model** (multi-label prediction of emotional tones)
3. Key phrases are extracted using YAKE.
4. Executive summary sentences are generated based on keyword overlap scoring.
5. Dashboard visualizes results with pie charts, bar charts, and downloadable CSVs.

---

## 🛠 Technologies Used

| Area | Tools / Libraries |
|:-----|:------------------|
| Language Model | Hugging Face Transformers |
| Training | PyTorch + Trainer API |
| Dataset Management | Hugging Face Datasets |
| UI / Frontend | Streamlit |
| Visualization | Matplotlib |
| Keyword Extraction | YAKE |
| Hosting | Hugging Face Hub + GitHub Codespaces |

---

## 🚀 Quickstart Guide

Clone the repository:

```bash
git clone https://github.com/goatley/sentiment-analysis-app.git
cd sentiment-analysis-app
pip install -r requirements.txt

Launch the dashboard:
streamlit run dashboard.py --server.port=8501 --server.headless true

🗂 Project Structure
sentiment-analysis-app/
├── dashboard.py               # Main Streamlit dashboard
├── scripts/                    # Fine-tuning and dataset scripts
│   ├── fine_tune_sentiment.py
│   ├── fine_tune_emotions.py
│   └── prepare_goemotions.py
├── models/                     # Fine-tuned model checkpoints
│   ├── sentiment_final/
│   └── emotion_final/
├── data/                       # Training datasets
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview (this file)
└── logs/                       # Training logs

📊 Model Details
Sentiment Model
Base: distilbert-base-uncased

Fine-tuned on Amazon review dataset (binary labels)

Achieved 85% test accuracy

Emotion Model
Base: distilbert-base-uncased

Fine-tuned on GoEmotions dataset (multi-label)

Predicts 27 emotions + neutral across real-world text

Both models are publicly available:

Sentiment Model on Hugging Face

🔗 Resources
[GitHub Repository](https://github.com/goatley/sentiment-analysis-app)
[Sentiment Model on Hugging Face](https://huggingface.co/goatley/sentiment-final-model)

🎯 Future Improvements
Fine-tune emotion model on domain-specific datasets

Integrate semantic topic modeling (e.g., BERTopic)

Add explainable AI (XAI) visualizations (e.g., LIME/SHAP)

Deploy dashboard to Hugging Face Spaces for public access

📚 Author
Keith Goatley
[Hugging Face Profile](https://huggingface.co/goatley)
[GitHub Profile](https://github.com/goatley)
