import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem import PorterStemmer

# create class objects
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

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

# Function to remove stop words
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    return text

def apply_stemmer(text):
    if isinstance(text, str):
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    return text

def apply_lemmatizer(text):
    if isinstance(text, str):
        doc = nlp(text)
        lemmatized_words = [token.lemma_ for token in doc]
        return ' '.join(lemmatized_words)
    return text
print(data[text_column][2])
# Apply remove_stopwords to the text column
data[text_column] = data[text_column].apply(remove_stopwords)
print(data[text_column][2])
data[text_column] = data[text_column].apply(apply_stemmer)
print(data[text_column][2])
data[text_column] = data[text_column].apply(apply_lemmatizer)
print(data[text_column][2])


