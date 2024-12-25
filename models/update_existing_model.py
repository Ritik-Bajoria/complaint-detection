from joblib import load, dump
import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report

# Load the existing model and vectorizer
model = load("./SVM_BOW_model.joblib")
vectorizer = load("./SVM_count_vectorizer.joblib")

# Load the dataset to be used 
data = pd.read_csv('Database/nepali-complaints.csv')

# Preprocess the text data to numerical data
data['label'] = data['label'].map({'complaint': 1, 'non-complaint': -1})

# Identify text column
text_column = 'text' 

# Text preprocessing function
def clean_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and excessive whitespace
        text = re.sub(r'[ \s*]+', ' ', text).strip()
        text = re.sub(r'[^A-Za-z0-9\u0900-\u097F\s:\n]', '', text)  # Keep English, Nepali (Devanagari), numbers, spaces, colons, and newlines
        text = re.sub(r'(\n\s*)+', '\n', text)  # Normalize multiple newlines to a single newline
    return text

# print("original sentence\t\t",data[text_column][200])
# Apply pre-processing functions to the text column
data[text_column] = data[text_column].apply(clean_text)
data[text_column] = data[text_column].fillna("")
# splitting data for training and for testing  in 8:2 ratio
X_train, X_test, y_train, y_test = train_test_split(data[text_column], data['label'], test_size=0.2, random_state=42)

# vectorize the dataset
X_train_count = vectorizer.transform(X_train.values)

# retrain the model
model.fit(X_train_count)

print("updating model..")
dump(model, "./SVM_BOW_model.joblib")
print("model updated\n")
print("updating vectorizer..")
dump(vectorizer,"./SVM_count_vectorizer.joblib")
print("vectorizer updated")

# Predict
# testing the model
X_test_count = vectorizer.transform(X_test)
y_pred = model.predict(X_test_count)

# show testing results
print(classification_report(y_test,y_pred))