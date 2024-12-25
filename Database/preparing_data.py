from langdetect import detect
import pandas as pd
import re
from autocorrect import Speller

# Create an instance of Speller
spell = Speller(lang='en')

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

# function to seperate rows in nepali or english
def detect_language_using_langdetect(text):
    try:
        detected_lang = detect(text)
        if detected_lang == 'en':  # 'ne' corresponds to Nepali
            return "English"
        else:
            return "Romanized Nepali"
    except Exception as e:
        return f"Error detecting language: {e}"

# Test the function
df = pd.read_excel("C://Users//Legion//Ritik//Desktop//Programming//Intern work//07-Intern//complaint detector//Database//CS_TICKETS_202412251154.xlsx")

df['ADDITIONAL_INFO'] = df['ADDITIONAL_INFO'].apply(clean_text)

english_column = []
nepali_column = []
english_label = []
nepali_label = []

for row in df['ADDITIONAL_INFO']:
    if detect_language_using_langdetect(row) == "English":
        english_column.append(row)
        english_label.append("complaint")
    else:
        nepali_column.append(row)
        nepali_label.append("complaint")
        
data_nepali = { 
    'text':nepali_column,
    'label':nepali_label
}
data_english = {
    'text':english_column,
    'label':english_label
}
df_nepali = pd.DataFrame(data_nepali)
df_english = pd.DataFrame(data_english)

df_nepali.to_csv("Database/nepali-complaints.csv")
df_english.to_csv("Database/english-complaints.csv")