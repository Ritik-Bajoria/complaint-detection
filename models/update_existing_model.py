from joblib import load, dump
import pandas as pd

# Load the model and vectorizer
model = load("./MultinomialNB_BOW_model.joblib")
vectorizer = load("./count_vectorizer.joblib")

def update_model(text):
    text_series = pd.Series(text)

    text_series_count = vectorizer.fit_transform(text_series)
    prediction = model.predict(text_series_count)

    model.fit(text_series_count,prediction)
    dump(model, "./MultinomialNB_BOW_model.joblib")
    