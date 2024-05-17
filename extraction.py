import nltk
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk import ne_chunk, pos_tag
import re
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Read data from CSV file
file_path = "Train.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Function to extract features from text
def extract_features(document):
    words = set(word_tokenize(document))
    features = {word: True for word in words}
    return features

labeled_data = [(extract_features(row['text']), row['label']) for _, row in data.iterrows()]

def extract_countries(text):
    entities = ne_chunk(pos_tag(word_tokenize(text)))
    countries = []
    country_pattern = re.compile(r'((?:[A-Z][a-z]+ ?)+)')


    for entity in entities:
        if hasattr(entity, 'label') and entity.label() == 'GPE':
            name = ' '.join(word for word, _ in entity.leaves())
            if country_pattern.match(name):
                countries.append(name)
    return countries

# Split data into training and testing sets using train_test_split
train_set, test_set = train_test_split(labeled_data, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set, estimator=nltk.probability.LaplaceProbDist)

# Test the classifier
print("Accuracy:", nltk.classify.accuracy(classifier, test_set))

# Read test data from file
file_path = "doc1"
with open(file_path, 'r') as file:
    test_data = file.read()

# Tokenize the test data into sentences
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

# Print sentences labeled as exclusion along with extracted countries
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
