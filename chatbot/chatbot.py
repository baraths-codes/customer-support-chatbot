import nltk
import string
import json
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier

# Download necessary NLTK resources
#nltk.download('punkt')
#nltk.download('stopwords')

# Load intents file (your JSON with intents, patterns, and responses)
with open('./data/intents.json', 'r') as file:
    intents = json.load(file)

# Preprocessing function to clean and tokenize text
stop_words = set(stopwords.words('english'))

def preprocess(text):
    words = word_tokenize(text.lower())  # Tokenize the sentence into words
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return words

# Extract features based on word presence
all_words = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = preprocess(pattern)
        all_words.extend(words)

# Get unique words (vocabulary)
all_words = list(set(all_words))

def extract_features(words):
    features = {}
    for word in all_words:
        features[word] = (word in words)  # Binary feature: 1 if word is in the sentence, else 0
    return features

# Prepare the training data
training_data = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = preprocess(pattern)
        features = extract_features(words)
        training_data.append((features, intent['tag']))

# Train the model (Naive Bayes Classifier)
classifier = NaiveBayesClassifier.train(training_data)

# Save the trained model using pickle
with open('chatbot_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

print("✅ Model trained and saved successfully!")

# Function to get the response based on the intent
def chatbot_response(user_input):
    words = preprocess(user_input)
    features = extract_features(words)
    
    # Predict the intent using the trained classifier
    predicted_tag = classifier.classify(features)
    
    # Find the response for the predicted tag
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            return response

# Start the chatbot (interactive)
print("Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    # Get the bot's response
    response = chatbot_response(user_input)
    print(f"Bot: {response}")