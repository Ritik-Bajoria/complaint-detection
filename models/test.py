from joblib import load, dump
import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer
from autocorrect import Speller
import spacy
from spacy.cli import download
from nltk.corpus import stopwords
from nltk.data import find
from sklearn.model_selection import train_test_split 

# Load the existing model and vectorizer
model = load("./MultinomialNB_BOW_model.joblib")
vectorizer = load("./count_vectorizer.joblib")

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
data = pd.read_csv('Database/newdata.csv')

# Preprocess the text data to numerical data
data['label'] = data['label'].map({'complaint': 1, 'non-complaint': 0})

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
complaints = apply_lemmatizer(correct_spell(apply_stemmer(remove_stopwords(clean_text(complaints)))))
non_complaints = apply_lemmatizer(correct_spell(apply_stemmer(remove_stopwords(clean_text(non_complaints)))))

# vectorize the lists
complaints_count = vectorizer.transform(complaints)
non_complaints_count = vectorizer.transform(non_complaints)

# make predictions for the lists
complaint_predictions = model.predict(complaints_count)
non_complaint_predictions = model.predict(non_complaints_count)

# print the predictions 
for prediction in complaint_predictions:
    print("complaint" if prediction == 1 else "non_complaint")  
for prediction in non_complaint_predictions:
    print("complaint" if prediction == 1 else "non_complaint")
