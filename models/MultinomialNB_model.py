import pandas as pd
import re
import spacy
from spacy.cli import download
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem import PorterStemmer
from autocorrect import Speller
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from joblib import dump

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
print(data['label'].value_counts())
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
        # pos_tags = [f"{token} {spacy.explain(token.pos_)}" for token in doc if token.pos_ not in ["SPACE","X","PUNCT"]]
        removed_unnecessary_text = [token.text for token in doc if spacy.explain(token.pos_) not in ["noun","proper noun", "pronoun", "numeral"]]
        print("||".join([f"{token.text} {spacy.explain(token.pos_)}" for token in doc if spacy.explain(token.pos_) not in ["noun","proper noun", "pronoun", "numeral"]]))
        return ' '.join(removed_unnecessary_text)
    return text

# split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(data[text_column], data['label'], test_size=0.2, random_state=10)

# print("original sentence\t\t",X_train)
# Apply pre-processing functions to the text column
X_train = X_train.apply(pos_tagging)
X_train = X_train.apply(clean_text)
# print("cleaned text\t\t",X_train)
X_train = X_train.apply(remove_stopwords)
# print("stopwords removed\t\t",X_train)
X_train = X_train.apply(apply_stemmer)
# print("words stemmed\t\t\t",X_train)
X_train = X_train.apply(correct_spell)
# print("spell corrected\t\t\t",X_train)
X_train = X_train.apply(apply_lemmatizer)
print("words lemmatized\t\t",X_train)

# create a new column in the dataframe to hold POS (Part of Speech) taggings 
# for each word in format {word} {POS Tag} | {word} {POS Tag}
data["pos_column"] = X_train.apply(pos_tagging)
print("pos tagged\t\t\t",data["pos_column"][10])

# initialize vectorizer
vectorizer = CountVectorizer(ngram_range=(1,1))
# fit the vectorizer to the training data and transform training data
X_train_count = vectorizer.fit_transform(X_train.values)

# initialize model
model = MultinomialNB()
# train the model
model.fit(X_train_count, y_train)

# saving the model and vectorizer 
dump(model,"MultinomialNB_BOW_model.joblib")
dump(vectorizer, "count_vectorizer.joblib")
print(len(vectorizer.get_feature_names_out()))
print(vectorizer.get_stop_words())

# testing the model
X_test_count = vectorizer.transform(X_test)
y_pred = model.predict(X_test_count)

# show testing results
print(classification_report(y_test,y_pred))

# making above model can simply be done using this pipeline too
# pipe = Pipeline([
#     ('vect', CountVectorizer()),
#     ('clf', MultinomialNB())
# ])
# pipe.fit(X_train,y_train)
# print(classification_report(y_test, pipe.predict(X_test)))

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
print(complaint_predictions)
non_complaint_predictions = model.predict(non_complaints_count)
print(non_complaint_predictions)

