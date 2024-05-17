import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import spacy

# Read data from CSV file
file_path = "Train.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1')

file_path = "doc1"
with open(file_path, 'r') as file:
    test_data = file.read()
def extract_features(document):
    words = set(word_tokenize(document))
    features = {word: True for word in words}
    return features

labeled_data = [(extract_features(row['text']), row['label']) for _, row in data.iterrows()]

nlp = spacy.load("en_core_web_sm")

# Function to extract countries from text using spacy
def extract_countries(text):
    doc = nlp(text)
    countries = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    return countries

def extract_countries(text):
    countries = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            countries.append(ent.text)
    return countries

train_set, test_set = train_test_split(labeled_data, test_size=0.2, random_state=42)

# Train
classifier = NaiveBayesClassifier.train(train_set, estimator=nltk.probability.LaplaceProbDist)

# Test
print("Accuracy:", nltk.classify.accuracy(classifier, test_set))

# Tokenize
sentences = sent_tokenize(test_data)

# Initialize a list to store sentences labeled as exclusion along with the extracted countries
exclusion_with_countries = []

# Classify each sentence and extract countries for sentences labeled as exclusion
for sentence in sentences:
    features = extract_features(sentence)
    classification = classifier.classify(features)
    if classification == 1:  # Exclusion label
        countries = extract_countries(sentence)
        exclusion_with_countries.append((sentence, countries))


for sentence, countries in exclusion_with_countries:
    print("Sentence:", sentence)
    print("Countries:", countries)
    print()

# Evaluate the classifier on the test set
actual_labels = [label for _, label in test_set]
predicted_labels = [classifier.classify(features) for features, _ in test_set]

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print("Accuracy:", accuracy)

# Generate classification report
# print("Classification Report:")
# print(classification_report(actual_labels, predicted_labels))