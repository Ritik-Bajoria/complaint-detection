# Import libraries
import numpy as np # to work with numpy arrays and vectors
import pandas as pd # pandas to work with excel data as data frames
import nltk 
from nltk.corpus import stopwords
from nltk.data import find
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from joblib import dump, load

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

# Apply to the text column
data[text_column] = data[text_column].apply(remove_stopwords)

# splitting data for training and for testing  in 8:2 ratio
X_train, X_test, y_train, y_test = train_test_split(data[text_column], data['label'], test_size=0.2, random_state=42)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train.values)  # Convert text to numeric features

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
model_file_path = 'logistic_regression_model.joblib' 
vectorizer_file_path = 'tfidf_vectorizer.joblib'
dump(model, model_file_path)
dump(vectorizer, vectorizer_file_path)

# Predict
y_pred = model.predict(vectorizer.transform(X_test.values))

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

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

# make predictions for the lists
complaint_predictions = model.predict(complaints_count)
non_complaint_predictions = model.predict(non_complaints_count)

# print the predictions 
for prediction in complaint_predictions:
    print("complaint" if prediction == 1 else "non_complaint")  
for prediction in non_complaint_predictions:
    print("complaint" if prediction == 1 else "non_complaint")