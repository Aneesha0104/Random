import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Flatten, Conv1D, GlobalMaxPooling1D, Dropout, Bidirectional
from nltk.tokenize import sent_tokenize

# Load data
data = pd.read_csv("augmented_data2.csv", encoding='ISO-8859-1')
with open("doc2", 'r') as file:
    test_data = file.read()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
data = data.dropna(subset=['text'])
data = data.dropna(subset=['label'])
# Function to extract countries using spaCy
def extract_countries(text):
    return [ent.text for ent in nlp(text).ents if ent.label_ == 'GPE']

# Preprocess text data
texts, labels = data['text'].values, data['label'].values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
def create_rnn_model():
    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, 128, input_length=100),
        Bidirectional(SimpleRNN(64, return_sequences=True)),
        SimpleRNN(64),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_mlp_model():
    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, 128, input_length=100),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model():
    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, 128, input_length=100),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate models
models = {
    "RNN": create_rnn_model(),
    "MLP": create_mlp_model(),
    "CNN": create_cnn_model()
}

for name, model in models.items():
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(f"{name} Test set accuracy:", accuracy_score(y_test, y_pred))

# Classify and extract countries
def classify_and_extract(model, sentences):
    results = []
    for sentence in sentences:
        seq = tokenizer.texts_to_sequences([sentence])
        padded_seq = pad_sequences(seq, maxlen=100)
        if (model.predict(padded_seq) > 0.5).astype("int32")[0][0] == 1:
            countries = extract_countries(sentence)
            if countries:
                results.append((sentence, countries))
    return results

# Tokenize test data into sentences
sentences = sent_tokenize(test_data)

# Print classification results for each model
for name, model in models.items():
    print(f"{name} Model Exclusion Sentences with Countries:")
    for sentence, countries in classify_and_extract(model, sentences):
        print("Sentence:", sentence)
        print("Countries:", countries)
        print()
