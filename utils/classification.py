from joblib import load
import pandas as pd

# Load the model and vectorizer
model = load("./MultinomialNB_BOW_model.joblib")
vectorizer = load("./count_vectorizer.joblib")

def classify(text):
    text_series = pd.Series(text)
    text_series_count = vectorizer.transform(text_series)
    prediction = model.predict(text_series_count)
    non_complaint_proba, complaint_proba = model.predict_proba(text_series_count)[0]
    print(complaint_proba, non_complaint_proba)
    if prediction == 1:
        return "Complaint", complaint_proba
    else:
        return "Not Complaint", non_complaint_proba