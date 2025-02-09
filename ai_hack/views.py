from django.shortcuts import render, redirect
from django.views.generic import CreateView
from django.http import HttpResponse
from .forms import CSVUploadForm
from django.http import HttpResponse
from .models import SentimentAnalysis

import csv
import joblib
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from django.shortcuts import render
from .models import SentimentAnalysis  # Import your model

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
    return ' '.join(tokens)


def HomeView(request):
    return render(request,template_name='home.html')

def AnalyzeView(request):
    results = []
    if request.method == "POST":
        user_text = request.POST.get('user_text')
        csv_file = request.FILES.get('csv_file')

        
        if user_text:
            # Save text input to database
            preprocessed_text = preprocess_text(user_text)

            text_vectorized = vectorizer.transform([preprocessed_text]).toarray()

            # Predict
            prediction = model.predict(text_vectorized)[0]
            sentiment_label = {0: "Positive", 1: "Neutral", 2: "Negative"}[prediction]

            # Save to database
            SentimentAnalysis.objects.create(text=user_text, predicted_sentiment=sentiment_label,
            cleaned_text = preprocessed_text,)

            results.append({"text": user_text, "prediction": sentiment_label})
               

        elif csv_file:
            # Process CSV file
            pass

    return render(request, 'analyze.html', {"results": results})

# MODEL_PATH = 'sentiment_model.pkl'
# model = joblib.load(MODEL_PATH)


def upload_csv(request):
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['file']
            decoded_file = csv_file.read().decode('utf-8').splitlines()
            reader = csv.reader(decoded_file)

            header = next(reader, None)

            comments = []
            for row in reader:
                comments.append(row[0])  # Assuming the first column is comments

            # Predict sentiments using the loaded model
            # predictions = model.predict(comments)

            results = []
            for i, comment in enumerate(comments):
                results.append({'comment': comment, 'predicted_sentiment': predictions[i]})

            # Render the results in a template
            return render(request, "results.html", {"results": results})
    else:
        form = CSVUploadForm()

    return render(request, "upload_csv.html", {"form": form})
