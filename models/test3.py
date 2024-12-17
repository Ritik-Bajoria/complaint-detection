from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the saved model
model = load('logistic_regression_model.joblib')

# Load the vectorizer used during training
vectorizer = load('tfidf_vectorizer.joblib')  # Assuming you saved the vectorizer similarly

# List of 50 statements to classify
statements = [
    "The product I received was defective and did not work as described.",
    "I am very happy with the fast delivery of my package.",
    "Customer support has not responded to my query for days.",
    "The user manual was clear and easy to understand.",
    "Why was my order canceled without informing me?",
    "Thank you for the excellent customer service!",
    "The item arrived two weeks late, and nobody updated me about the delay.",
    "I love the new design of your website, very user-friendly!",
    "The product description was misleading; this is not what I ordered.",
    "The return process was smooth and hassle-free.",
    "My subscription was charged twice, and I need a refund.",
    "The quality of the product is excellent, very satisfied.",
    "The support team resolved my issue quickly, great job!",
    "The package was damaged when it arrived.",
    "I appreciate the effort put into packaging the item securely.",
    "My account was locked without any valid reason.",
    "Fast and reliable service, highly recommend.",
    "The payment system is not working; I cannot complete my purchase.",
    "Amazing discounts and offers, I will shop here again!",
    "I received a completely different item from what I ordered.",
    "The refund process is taking way too long.",
    "The instructions were incomplete, making it hard to set up the product.",
    "Customer care was polite and resolved my issue quickly.",
    "I am disappointed with the poor quality of the item.",
    "I will definitely recommend this to my friends.",
    "The product stopped working after just one use.",
    "The app crashes frequently, please fix it.",
    "Everything went smoothly with my order, very satisfied.",
    "I had to wait an hour to speak to customer service.",
    "The product was exactly as described, good experience.",
    "My order tracking details are incorrect.",
    "This is the best shopping experience I have had online!",
    "The replacement item I received is also defective.",
    "Thank you for the prompt delivery.",
    "The food I ordered was stale and inedible.",
    "The packaging was neat, and everything was intact.",
    "I was charged for a service I did not use.",
    "The delivery was on time, and the product was well-packed.",
    "Nobody informed me about the delay in my order.",
    "I love the quality of the material used in this product.",
    "The customer support agent was rude and unhelpful.",
    "The product arrived earlier than expected, excellent service!",
    "I need assistance with setting up the software I purchased.",
    "The website is slow and crashes frequently.",
    "I am impressed with the durability of the product.",
    "The size of the product does not match the description.",
    "Overall, I had a pleasant shopping experience.",
    "The instructions were missing from the package.",
    "The wrong item was delivered to my address."
]

# Corresponding labels for the statements
labels = [
    "complaint", "non-complaint", "complaint", "non-complaint",
    "complaint", "non-complaint", "complaint", "non-complaint",
    "complaint", "non-complaint", "complaint", "non-complaint",
    "non-complaint", "complaint", "non-complaint", "complaint",
    "non-complaint", "complaint", "non-complaint", "complaint",
    "complaint", "complaint", "non-complaint", "complaint",
    "non-complaint", "complaint", "complaint", "non-complaint",
    "complaint", "non-complaint", "complaint", "non-complaint",
    "complaint", "non-complaint", "complaint", "non-complaint",
    "complaint", "non-complaint", "complaint", "non-complaint",
    "non-complaint", "complaint", "non-complaint", "complaint",
    "non-complaint", "complaint", "non-complaint", "complaint"
]

# Function to remove stop words
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    return text

# Initialize counters
successful_tests = 0

# Process and classify each statement
for i, (text, actual_label) in enumerate(zip(statements, labels)):
    processed_text = remove_stopwords(text)  # Remove stop words
    text_vectorized = vectorizer.transform([processed_text])  # Vectorize the text
    prediction = model.predict(text_vectorized)  # Predict

    # Map numeric prediction to label
    predicted_label = "complaint" if prediction == 1 else "non-complaint"

    # Check if the prediction matches the actual label
    if predicted_label == actual_label:
        successful_tests += 1

    # Print the result
    print(f"{i+1}. {text} -> Predicted: {predicted_label}, Actual: {actual_label}")

# Display success rate
success_rate = (successful_tests / len(statements)) * 100
print(f"\nTotal Successful Tests: {successful_tests}/{len(statements)}")
print(f"Success Rate: {success_rate:.2f}%")
