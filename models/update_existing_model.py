from joblib import load, dump
import pandas as pd

# Load the existing model and vectorizer
model = load("./MultinomialNB_BOW_model.joblib")
vectorizer = load("./count_vectorizer.joblib")

def update_model(text):

    # load new dataset
    data = pd.read_csv("newdata.csv")

    # vectorize the dataset
    text_series_fit_count = vectorizer.fit_transform(data["text"])
    
    # retrain the model
    model.fit(text_series_fit_count, data["label"])
    
    print("updating model..")
    dump(model, "./MultinomialNB_BOW_model.joblib")
    print("model updated")
    