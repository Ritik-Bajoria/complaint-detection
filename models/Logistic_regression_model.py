# Import libraries
import numpy as np # to work with numpy arrays and vectors
import pandas as pd # pandas to work with excel data as data frames
import re
import spacy
from spacy.cli import download
import nltk
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem import PorterStemmer
from autocorrect import Speller
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from joblib import dump, load

# Create an instance of the Porterstemmer
stemmer = PorterStemmer()

# Adjust display width to show full rows
pd.set_option('display.width', None)  # None means no width limit
pd.set_option('display.max_colwidth', None)  # Show full column content

# Create an instance of Speller
spell = Speller(lang='en')

# Check if the model is already installed, otherwise install it
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# custom changes to nlp pipelines
ar = nlp.get_pipe('attribute_ruler')
ar.add([[{"TEXT":"n’t"}],[{"TEXT":"n't"}]],{"LEMMA":"not"})
ar.add([[{"TEXT":"bro"}],[{"TEXT":"brah"}]],{"LEMMA":"brother"})
ar.add([[{"TEXT":"plz"}]],{"LEMMA":"please"})
ar.add([[{"TEXT":"thnx"}],[{"TEXT":"ty"}],[{"TEXT":"thx"}],[{"TEXT":"thank"}]],{"LEMMA":"thank"})

# Check if 'stopwords' resource exists
try:
    find('corpora/stopwords.zip')  # Check if stopwords are downloaded
except LookupError:
    nltk.download('stopwords')

# Load the dataset to be used 
data = pd.read_csv('C:/Users/Legion/Ritik/Desktop/Programming/Intern work/07-Intern/complaint detector/Database/mydata_balanced.csv')

# Preprocess the text data to numerical data
data['label'] = data['label'].map({'complaint': 1, 'non-complaint': 0})

# Identify text column
text_column = 'text' 

# splitting data for training and for testing  in 8:2 ratio
X_train, X_test, y_train, y_test = train_test_split(data[text_column], data['label'], test_size=0.2)

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

# Function to remove stop words
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)
    return text

def apply_stemmer(text):
    if isinstance(text, str):
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    return text

def correct_spell(text):
    if isinstance(text, str):
        words = text.split()
        corrected_words = [spell(word) for word in words]
        return ' '.join(corrected_words)
    return text
    
def apply_lemmatizer(text):
    if isinstance(text, str):
        doc = nlp(text)
        lemmatized_words = [token.lemma_ for token in doc]
        return ' '.join(lemmatized_words)
    return text

def pos_tagging(text):
    if isinstance(text, str):
        doc = nlp(text)
        pos_tags = [f"{token} {spacy.explain(token.pos_)}" for token in doc if token.pos_ not in ["SPACE","X","PUNCT"]]
        return ' | '.join(pos_tags)
    return text

# print("original sentence\t\t",X_train[1])
# Apply pre-processing functions to the text column
X_train = X_train.apply(clean_text)
# print("cleaned text\t\t",X_train[1])
X_train = X_train.apply(remove_stopwords)
# print("stopwords removed\t\t",X_train[1])
X_train = X_train.apply(apply_stemmer)
# print("words stemmed\t\t\t",X_train[1])
X_train = X_train.apply(correct_spell)
# print("spell corrected\t\t\t",X_train[1])
X_train = X_train.apply(apply_lemmatizer)
# print("words lemmatized\t\t",X_train[1])

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)  # Convert training text to numeric features

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
model_file_path = 'logistic_regression_tfidf_model.joblib'
vectorizer_file_path = 'tfidf_vectorizer.joblib'
dump(model, model_file_path)
dump(vectorizer, vectorizer_file_path)

# Predict on test data
X_test_tfidf = vectorizer.transform(X_test)  # Transform test data into numeric features
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(classification_report(y_test, y_pred))

# testing using lists
complaints = [
    "The product was missing several parts when delivered.",
    "I was billed for a service I never subscribed to.",
    "The website crashes every time I try to place an order.",
    "Customer service never responded to my refund request.",
    "The delivery person was rude and unprofessional."
]
non_complaints = [
    "The product quality exceeded my expectations.",
    "I received my order on time, perfectly packed.",
    "The support team was friendly and very helpful.",
    "I am impressed with how easy it was to navigate your website.",
    "Thank you for resolving my issue so quickly."
]
# pre-process the lists
# complaints = apply_lemmatizer(correct_spell(apply_stemmer(remove_stopwords(clean_text(complaints)))))
# non_complaints = apply_lemmatizer(correct_spell(apply_stemmer(remove_stopwords(clean_text(non_complaints)))))

# vectorize the lists
complaints_count = vectorizer.transform(complaints)
non_complaints_count = vectorizer.transform(non_complaints)
print(complaints_count)
rows, cols = complaints_count.nonzero()
for i in range(len(rows)):
    print(f"Row: {rows[i]}, Column: {vectorizer.get_feature_names_out()[cols[i]]}")
# make predictions for the lists
complaint_predictions = model.predict(complaints_count)
non_complaint_predictions = model.predict(non_complaints_count)

# print the predictions 
for prediction in complaint_predictions:
    print("complaint" if prediction == 1 else "non_complaint")  
for prediction in non_complaint_predictions:
    print("complaint" if prediction == 1 else "non_complaint")