from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the saved model
model = load('logistic_regression_model.joblib')

# Load the vectorizer used during training (you should save it as well)
vectorizer = load('tfidf_vectorizer.joblib')  # Assuming you saved the vectorizer similarly

# Text you want to classify
text = "I just finished reading a great book about personal development"

# Function to remove stop words
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    return text
text = remove_stopwords(text)
print(text)

# Vectorize the input text using the same vectorizer
text_vectorized = vectorizer.transform([text])

# Predict using the loaded model
prediction = model.predict(text_vectorized)

# Display the result
if prediction == 1:
    print("Complaint")
else:
    print("Non-Complaint")
