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
data = pd.read_csv('C:/Users/Legion/Ritik/Desktop/Programming/Intern work/07-Intern/complaint detector/Database/mydata.csv')

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

# print("original sentence\t\t",data[text_column][200])
# Apply pre-processing functions to the text column
data[text_column] = data[text_column].apply(clean_text)
# print("cleaned text\t\t",data[text_column][200])
data[text_column] = data[text_column].apply(remove_stopwords)
# print("stopwords removed\t\t",data[text_column][200])
data[text_column] = data[text_column].apply(apply_stemmer)
# print("words stemmed\t\t\t",data[text_column][200])
data[text_column] = data[text_column].apply(correct_spell)
# print("spell corrected\t\t\t",data[text_column][200])
data[text_column] = data[text_column].apply(apply_lemmatizer)
# print("words lemmatized\t\t",data[text_column][200])

# create a new column in the dataframe to hold POS (Part of Speech) taggings 
# for each word in format {word} {POS Tag} | {word} {POS Tag}
data["pos_column"] = data[text_column].apply(pos_tagging)
# print("pos tagged\t\t\t",data["pos_column"][200])

# vectorize the dataset
text_series_fit_count = vectorizer.fit_transform(data[text_column])

# retrain the model
model.fit(text_series_fit_count, data["label"])

print("updating model..")
dump(model, "./MultinomialNB_BOW_model.joblib")
print("model updated")
