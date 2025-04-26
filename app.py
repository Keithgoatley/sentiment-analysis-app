import gradio as gr
import joblib

# Load your trained sentiment analysis model
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Function for predicting sentiment
def predict_sentiment(text):
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return sentiment

# Create Gradio web interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter your review here..."),
    outputs="text",
    title="Custom Sentiment Analyzer",
    description="This app uses your own trained AI model to classify text sentiment."
)

interface.launch()
