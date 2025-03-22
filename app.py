import os
import json
import numpy as np
import re
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Initialize Flask app
app = Flask(__name__)

# Load dataset
with open("intents.json", "r") as file:
    data = json.load(file)["intents"]

symptoms = [" ".join(intent["symptoms"]) for intent in data]
diseases = [intent["disease"] for intent in data]
guidance = {intent["disease"]: intent["guidance"] for intent in data}

# Text preprocessing
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9, ]", "", text)
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words])

# Vectorization & Label Encoding
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(symptoms).toarray()

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(diseases)

# Build Neural Network
model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(len(set(y_train)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, verbose=1)

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_message = request.json.get("message", "").lower()
    
    # Match symptoms from dataset
    matched_disease = None
    for intent in data:
        for symptom in intent["symptoms"]:
            if symptom in user_message:  # Simple substring matching
                matched_disease = intent["disease"]
                guidance_text = intent["guidance"]
                break  # Stop when first match is found
        if matched_disease:
            break  # Stop checking further diseases
    
    # If no match, return unknown response
    if not matched_disease:
        return jsonify({"disease": "Unknown", "guidance": "Consult a doctor for better diagnosis."})

    return jsonify({"disease": matched_disease, "guidance": guidance_text})


if __name__ == "__main__":
    app.run(debug=True)
