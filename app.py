# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
nb_classifier = joblib.load('nb_classifier.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define a route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    features = tfidf_vectorizer.transform([input_text])
    prediction = nb_classifier.predict(features)[0]
    probability = max(nb_classifier.predict_proba(features)[0])

    return jsonify({'prediction': prediction, 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)
