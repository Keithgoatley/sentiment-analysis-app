import gradio as gr
from transformers import pipeline

# Load HuggingFace sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis')

# Define the analysis function
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    return f"Sentiment: {sentiment}, Confidence: {confidence:.2f}"

# Gradio interface setup
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="AI-powered Sentiment Analyzer",
    description="Type any text and instantly see its sentiment."
)

# Launch app
interface.launch()
