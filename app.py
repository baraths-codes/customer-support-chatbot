import nltk_setup
from flask import Flask, request, jsonify
import json
import random
import string
import nltk
import numpy as np
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# NLTK resources (make sure downloaded at least once)
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# Load intents
with open('./data/intents.json') as file:
    data = json.load(file)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Prepare training data
sentences = []
labels = []
tags = []
for intent in data['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        sentences.append(preprocess(pattern))
        labels.append(intent['tag'])

tags = sorted(set(tags))
tag_to_index = {tag: i for i, tag in enumerate(tags)}
index_to_tag = {i: tag for tag, i in tag_to_index.items()}
y_train = np.array([tag_to_index[label] for label in labels])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(sentences)

model = LogisticRegression()
model.fit(X_train, y_train)

# Flask setup
app = Flask(__name__)
CORS(app)  # This allows your frontend to access Flask API
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    processed_input = preprocess(user_input)
    X_test = vectorizer.transform([processed_input])
    prediction = model.predict(X_test)[0]
    tag = index_to_tag[prediction]

    for intent in data['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return jsonify({"response": response})
    return jsonify({"response": "I'm not sure how to help with that."})

if __name__ == "__main__":
    app.run(debug=True)
