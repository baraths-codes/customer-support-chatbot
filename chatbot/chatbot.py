import json
import random
import string
import nltk
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources (only once)
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# Load intents file
with open('./data/intents.json') as file:
    data = json.load(file)

# Preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)  # Return as string for TF-IDF

# Prepare training data
sentences = []
labels = []
tags = []
for intent in data['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        sentences.append(preprocess(pattern))
        labels.append(intent['tag'])

# Encode labels
tags = sorted(set(tags))
tag_to_index = {tag: i for i, tag in enumerate(tags)}
index_to_tag = {i: tag for tag, i in tag_to_index.items()}
y_train = np.array([tag_to_index[label] for label in labels])

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(sentences)

# Train classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Chat function
def chat():
    print("🤖 Bot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() == 'quit':
            print("🤖 Bot: Goodbye!")
            break
        processed_input = preprocess(user_input)
        X_test = vectorizer.transform([processed_input])
        prediction = model.predict(X_test)[0]
        tag = index_to_tag[prediction]

        for intent in data['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                print("🤖 Bot:", response)
                break

# Run the chatbot
if __name__ == "__main__":
    chat()
