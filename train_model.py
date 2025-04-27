import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load prepared dataset
df = pd.read_csv("amazon_reviews.csv")

# Verify data loaded correctly
print(df.head())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train Logistic Regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_vec, y_train)

# Evaluate model performance on test data
y_pred = classifier.predict(X_test_vec)
report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
print("Classification Report:\n", report)

# Save trained model and vectorizer for later use
joblib.dump(classifier, "sentiment_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
