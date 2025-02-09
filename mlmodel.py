# sentiment_model.py (example, train a model)

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline
import nltk
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score



# Sample data (you'll replace this with your actual dataset)
df = pd.read_csv("hotel_reviews.csv")

sentiment_map = {"positive": 0, "neutral": 1, "negative": 2}
df["label_sentiment"] = df["label"].map(sentiment_map)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Fill any NaN values in 'Comments' column with an empty string and ensure it's a string
#df['Comments'] = df['comment'].fillna('').astype(str)

# Preprocessing Steps
# 1. Convert to lowercase
df['comment_cleaned'] = df['comment'].str.lower()

# 2. Remove special characters, punctuation, and numbers
df['comment_cleaned'] = df['comment_cleaned'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# 3. Tokenize the comments
df['comment_tokenized'] = df['comment_cleaned'].apply(word_tokenize)

# 4. Remove stopwords
df['comment_no_stopwords'] = df['comment_tokenized'].apply(
    lambda tokens: [word for word in tokens if word not in stop_words]
)

# 5. Lemmatize the tokens
df['comment_lemmatized'] = df['comment_no_stopwords'].apply(
    lambda tokens: [lemmatizer.lemmatize(word) for word in tokens]
)

# 6. Join the tokens back into a string
df['comment_preprocessed'] = df['comment_lemmatized'].apply(lambda tokens: ' '.join(tokens))

# Display the first few rows of the preprocessed data
print(df[['comment', 'comment_preprocessed']].head())



# Assuming `df` is your DataFrame
# Columns: ['comment_preprocessed', 'label_sentiment']

# Step 1: Vectorize the `comment_preprocessed` column using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features for dimensionality
X = vectorizer.fit_transform(df['comment_preprocessed']).toarray()

# Step 2: Prepare the target column
y = df['label_sentiment']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Train a Random Forest Classifier
clf = RandomForestClassifier(
    n_estimators=200,       # Increase estimators for better results
    max_depth=20,           # Limit tree depth to avoid overfitting
    random_state=42,
    n_jobs=-1,              # Use all CPU cores for faster training
    class_weight='balanced' # Handle potential class imbalance
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Step 6: Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(clf, 'sentiment_model.pkl')
