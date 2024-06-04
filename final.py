import numpy as np
import pandas as pd
import spacy
from nltk import sent_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SimpleRNN, Dense, Dropout, Bidirectional

# Load data
data = pd.read_csv("augmented_data2_og.csv", encoding='ISO-8859-1')
with open("doc2", 'r') as file:
    test_data = file.read()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Remove rows with missing text or label
data = data.dropna(subset=['text', 'label'])

# Function to extract countries using spaCy
def extract_countries(text):
    return [ent.text for ent in nlp(text).ents if ent.label_ == 'GPE']

# Preprocess text data
texts, labels = data['text'].values, data['label'].values

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=100)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "LSTM": Sequential([
        Embedding(len(tokenizer.word_index) + 1, 128, input_length=100),
        LSTM(64, return_sequences=True, kernel_regularizer='l2'),
        LSTM(64, kernel_regularizer='l2'),
        Dense(128, activation='relu', kernel_regularizer='l1'),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer='l1'),
        Dropout(0.5),
        Dense(1, activation='sigmoid', kernel_regularizer='l2')
    ]),
    "RNN": Sequential([
        Embedding(len(tokenizer.word_index) + 1, 128, input_length=100),
        Bidirectional(SimpleRNN(64, return_sequences=True)),
        SimpleRNN(64),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ]),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB()
}

# Train and evaluate models
for name, model in models.items():
    if name in ["LSTM", "RNN"]:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, verbose=0)
    else:
        vectorizer = CountVectorizer()
        X_vec = vectorizer.fit_transform(texts)
        X_train_vec, X_test_vec, y_train_vec, y_test_vec = train_test_split(X_vec, y, test_size=0.2, random_state=42)
        model.fit(X_train_vec, y_train_vec)

# Function to classify and extract countries
def classify_and_extract(model, sentences):
    for sentence in sentences:
        if "LSTM" in model.name or "RNN" in model.name:
            seq = tokenizer.texts_to_sequences([sentence])
            padded_seq = pad_sequences(seq, maxlen=100)
            prediction = (model.predict(padded_seq, verbose=0) > 0.5).astype("int32")[0][0]
        else:
            features = vectorizer.transform([sentence])
            prediction = model.predict(features)[0]

        if prediction == 1:
            countries = extract_countries(sentence)
            if countries:
                print(f"Sentence: {sentence}")
                print(f"Countries: {countries}")

# Tokenize test data into sentences
sentences = sent_tokenize(test_data)

# Classify and extract countries
for name, model in models.items():
    print(f"\n{name} Model Exclusion Sentences with Countries:")
    classify_and_extract(model, sentences)

# Pickle all the models
import pickle

model_files = {}
for name, model in models.items():
    with open(f"{name.lower()}_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    model_files[name] = f"{name.lower()}_model.pkl"
