from joblib import load
import pandas as pd
import numpy as np

# Load the model and vectorizer
model = load("./SVM_BOW_model.joblib")
vectorizer = load("./SVM_count_vectorizer.joblib")

def classify(text):
    text_series = pd.Series(text)
    text_series_count = vectorizer.transform(text_series)
    prediction = model.predict(text_series_count)
    # # Get decision scores
    # decision_scores = model.decision_function(text_series_count)
    # complaint_proba = 1 / (1 + np.exp(-decision_scores))
    # non_complaint_proba = 1 - complaint_proba
    if prediction == 1:
        return "Complaint"
    else:
        return "Not Complaint"