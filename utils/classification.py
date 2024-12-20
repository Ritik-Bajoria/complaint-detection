from joblib import load
import pandas as pd

# Load the model and vectorizer
model = load("./MultinomialNB_BOW_model.joblib")
vectorizer = load("./count_vectorizer.joblib")

def classify(text):
    text_series = pd.Series(text)
    text_series_count = vectorizer.transform(text_series)
    prediction = model.predict(text_series_count)
    if prediction == 1:
        return "Complaint"
    else:
        return "Not Complaint"