import streamlit as st
import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the saved BERT model
model_path = "./bert_model"
bert_model = BertForSequenceClassification.from_pretrained(model_path)

# Load the saved tokenizer
tokenizer_path = "./bert_tokenizer_model"
bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Load the RNN and LSTM models
rnn_model = tf.keras.models.load_model('rnn_model.h5')
lstm_model = tf.keras.models.load_model('lstm_model.h5')

# Load the Naive Bayes and Random Forest models
with open('naive_bayes_model.h5', 'rb') as handle:
    naive_bayes_model = pickle.load(handle)

with open('random_forest_model.h5', 'rb') as handle:
    random_forest_model = pickle.load(handle)

# Load the tokenizers for RNN and LSTM
with open('tokenizer_rnn.pkl', 'rb') as handle:
    tokenizer_rnn = pickle.load(handle)

with open('tokenizer_lstm.pickle', 'rb') as handle:
    tokenizer_lstm = pickle.load(handle)

# Load the CountVectorizer for Naive Bayes and Random Forest
with open('vectorizer_nb_rf.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to extract countries using spaCy
def extract_countries(text):
    return [ent.text for ent in nlp(text).ents if ent.label_ == 'GPE']

# Function to classify and extract countries using RNN or LSTM
def classify_and_extract_nn(text, model, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=100)
    prediction = (model.predict(padded_seq) > 0.5).astype("int32")[0][0]
    if prediction == 1:
        countries = extract_countries(text)
        return countries
    else:
        return []

# Function to classify and extract countries using Naive Bayes or Random Forest
def classify_and_extract_ml(text, model, vectorizer):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    if prediction == 1:
        countries = extract_countries(text)
        return countries
    else:
        return []

# Function to classify and extract countries using BERT
def classify_and_extract_bert(text):
    encodings = bert_tokenizer(text, truncation=True, padding=True, max_length=100, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**encodings)
    predictions = torch.sigmoid(outputs.logits).numpy().flatten()
    countries = extract_countries(text) if predictions[0] > 0.5 else []
    return countries

# Streamlit app
st.title("Country Extraction from Text")

model_choice = st.selectbox("Choose a model:", ["Naive Bayes", "Random Forest", "RNN", "LSTM", "BERT"])

user_input = st.text_area("Enter a sentence or paragraph:")

if st.button("Extract"):
    if user_input:
        sentences = user_input.split('.')
        result = []

        for sentence in sentences:
            if model_choice == "RNN":
                model = rnn_model
                tokenizer = tokenizer_rnn
                countries = classify_and_extract_nn(sentence.strip(), model, tokenizer)
            elif model_choice == "LSTM":
                model = lstm_model
                tokenizer = tokenizer_lstm
                countries = classify_and_extract_nn(sentence.strip(), model, tokenizer)
            elif model_choice == "Naive Bayes":
                model = naive_bayes_model
                countries = classify_and_extract_ml(sentence.strip(), model, vectorizer)
            elif model_choice == "Random Forest":
                model = random_forest_model
                countries = classify_and_extract_ml(sentence.strip(), model, vectorizer)
            elif model_choice == "BERT":
                countries = classify_and_extract_bert(sentence.strip())

            if countries:
                result.append((sentence, countries))

        if result:
            st.write("**Extracted exclusion statements and Countries:**")
            for sentence, countries in result:
                st.write(f"Sentence: {sentence}")
                st.write(f"Countries: {', '.join(countries)}")
        else:
            st.write("No  exclusion statements and countries were detected in the text.")
    else:
        st.write("Please enter some text.")
