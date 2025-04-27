---
title: "Sentiment & Emotion Analytics Dashboard"
author: "Keith Goatley"
date: "April 2025"
geometry: margin=1in
fontsize: 11pt
---

# Executive Summary  
This document presents the design, implementation, and evaluation of an advanced NLP dashboard built to analyze textual reviews. It demonstrates applied skills in deep learning, data engineering, and full-stack deployment.

# 1. Introduction & Motivation  
The ability to quantify public opinion and emotional nuance in text is critical for customer insights. This project—part of the MSc Artificial Intelligence application—develops a dual-model pipeline that classifies sentiment (positive/negative) and fine-grained emotion (27 categories) then visualizes results in an interactive dashboard.

# 2. System Design & Architecture  
The system consists of a Streamlit frontend that calls two fine-tuned DistilBERT pipelines (one for sentiment analysis and one for emotion classification), performs YAKE keyword extraction, and runs a simple extractive summarizer. Results are visualized with Matplotlib and offered for download as CSV.

# 3. Model Training Details  
## 3.1 Sentiment Model  
- **Base checkpoint:** `distilbert-base-uncased`  
- **Dataset:** 4 million Amazon reviews (balanced positive/negative)  
- **Fine-tuning:** 3 epochs, linear learning-rate decay schedule  
- **Evaluation:** 85 % accuracy on held-out test split  

## 3.2 Emotion Model  
- **Base checkpoint:** `distilbert-base-uncased`  
- **Dataset:** GoEmotions (43 k annotated comments, 27 emotions + neutral)  
- **Fine-tuning:** 3 epochs, multi-label Trainer API  
- **Evaluation:** 0.85 micro-F1 on test split  

# 4. Dashboard Implementation  
- **Key components:**  
  1. Load Hugging Face pipelines for sentiment and emotion  
  2. Batch inference on user‐provided texts  
  3. YAKE keyword extraction for top phrases  
  4. Keyword-driven extractive executive summary  
  5. Visualizations (pie charts for sentiment, bar charts for emotion) and CSV download  
- **Source code:** [`dashboard.py`](dashboard.py)

# 5. Results & Evaluation  
| Metric             | Sentiment | Emotion (micro-F1) |
|--------------------|:---------:|:-----------------:|
| Test Accuracy/F1   |   85 %    |       0.85        |

<details>
<summary>Sample Dashboard Screenshot</summary>
  
![Dashboard Screenshot](assets/dashboard_screenshot.png)
</details>

# 6. Future Work  
- Domain-specific re-training of the emotion model  
- Integration of semantic topic modeling (e.g., BERTopic)  
- Explainable AI overlays (LIME, SHAP)  
- Public deployment to Hugging Face Spaces  

# 7. References & Resources  
- GitHub repo: [https://github.com/Keithgoatley/sentiment-analysis-app](https://github.com/Keithgoatley/sentiment-analysis-app)  
- Sentiment model: [https://huggingface.co/goatley/sentiment-final-model](https://huggingface.co/goatley/sentiment-final-model)  
- GoEmotions paper: *Demszky et al. (2020)*  
- YAKE: Campos et al. (2020)  

---

*Report generated with Pandoc & Markdown.*  
