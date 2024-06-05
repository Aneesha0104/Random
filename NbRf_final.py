import numpy as np
import pandas as pd
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import pickle

# Read data from CSV file
file_path = "augmented_data2_og.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1')

file_path = "doc2"
with open(file_path, 'r') as file:
    test_data = file.read()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract countries from text using spaCy
def extract_countries(text):
    doc = nlp(text)
    countries = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    return countries

# Remove rows with missing text or label
data.dropna(inplace=True)

# Use CountVectorizer to transform the text data into feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers with best hyperparameters
naive_bayes_params = {
    'alpha': 0.1
}
random_forest_params = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

naive_bayes = MultinomialNB(**naive_bayes_params)
random_forest = RandomForestClassifier(**random_forest_params)

classifiers = {
    'Naive Bayes': naive_bayes,
    'Random Forest': random_forest
}

# Function to evaluate and print results for each classifier
def evaluate_classifier(name, classifier, X_train, y_train, X_test, y_test):
    print(f"Classifier: {name}")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test set accuracy:", accuracy)

    # Tokenize the test data into sentences
    sentences = sent_tokenize(test_data)

    # Initialize a list to store sentences labeled as exclusion along with the extracted countries
    exclusion_with_countries = []

    # Classify each sentence and extract countries for sentences labeled as exclusion
    for sentence in sentences:
        features = vectorizer.transform([sentence])
        classification = classifier.predict(features)[0]
        if classification == 1:  # Exclusion labels
            countries = extract_countries(sentence)
            if countries:  # Only include sentences with country names
                exclusion_with_countries.append((sentence, countries))

    # Print sentences labeled as exclusion along with extracted countries
    for sentence, countries in exclusion_with_countries:
        print("Sentence:", sentence)
        print("Countries:", countries)
        print()

    print("\n" + "="*80 + "\n")

# Evaluate each classifier
for name, classifier in classifiers.items():
    evaluate_classifier(name, classifier, X_train, y_train, X_test, y_test)

# Save the models
model_filenames = {
    'Naive Bayes': 'naive_bayes_model.h5',
    'Random Forest': 'random_forest_model.h5'
}

for name, classifier in classifiers.items():
    with open(model_filenames[name], 'wb') as file:
        pickle.dump(classifier, file)

# Save the CountVectorizer
with open('vectorizer_nb_rf.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Models and vectorizer saved successfully.")
