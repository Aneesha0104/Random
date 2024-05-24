import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.regularizers import l1, l2
from nltk.tokenize import sent_tokenize

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
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with L1 and L2 regularization
def create_lstm_model():
    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, 128, input_length=100),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
        LSTM(64, kernel_regularizer=l2(0.01)),
        Dense(128, activation='relu', kernel_regularizer=l1(0.01)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l1(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_bilstm_model():
    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, 128, input_length=100),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))),
        LSTM(64, kernel_regularizer=l2(0.01)),
        Dense(128, activation='relu', kernel_regularizer=l1(0.01)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l1(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model():
    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, 128, input_length=100),
        GRU(64, return_sequences=True, kernel_regularizer=l2(0.01)),
        GRU(64, kernel_regularizer=l2(0.01)),
        Dense(128, activation='relu', kernel_regularizer=l1(0.01)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l1(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate models
models = {
    "LSTM": create_lstm_model(),
    "BiLSTM": create_bilstm_model(),
    "GRU": create_gru_model()
}

for name, model in models.items():
    print(f"Training {name} model...")
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, verbose=0)
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype("int32")
    print(f"{name} Test set accuracy:", accuracy_score(y_test, y_pred))

# Classify and extract countries
def classify_and_extract(model, sentences):
    results = set()
    for sentence in sentences:
        seq = tokenizer.texts_to_sequences([sentence])
        padded_seq = pad_sequences(seq, maxlen=100)
        if (model.predict(padded_seq, verbose=0) > 0.5).astype("int32")[0][0] == 1:
            countries = extract_countries(sentence)
            if countries:
                results.add((sentence, tuple(countries)))
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
