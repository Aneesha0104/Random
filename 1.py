import nltk
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.classify import NaiveBayesClassifier

data = [
    ("All Documents provided to evidence no Iran vessel involved", 1),
    ("Documentation confirming no Iran vessel involved in trading.", 1),
    ("Proof of no Iran vessel involvement through submitted documents", 1),
    ("This is a random statement.", 0),
    ("Another non-exclusion statement.", 0),
    ("This is definitely not about vessels.", 0),
    ("Records showing no Syrian vessel participation.", 1),
    ("Verification documents ensure no North Korean vessel engaged.", 1),
    ("Confirmed: no vessels from Cuba were used.", 1),
    ("No evidence of any Libyan vessel involvement found.", 1),
    ("Reports confirming absence of Sudanese vessels in the trade.", 1),
    ("Official documents prove no Russian vessel engagement.", 1),
    ("Ensuring no Venezuelan vessel was utilized.", 1),
    ("No Myanmar vessel has been documented in the activities.", 1),
    ("Unrelated statement without any specific details.", 0),
    ("A completely irrelevant sentence.", 0),
    ("This line does not pertain to the subject.", 0),
    ("Irrelevant information with no context.", 0),
    ("No mention of vessel or trade.", 0),
    ("Random text without meaningful content.", 0),
    ("This paragraph is unrelated to exclusion.", 0),
    ("No Iranian vessels were found in the recent inspections.", 1),
    ("Our data confirms no involvement of Iranian vessels.", 1),
    ("Documents have been provided to confirm no Iranian vessel usage.", 1),
    ("A thorough review ensures no participation of Iranian vessels.", 1),
    ("Reports verify the absence of Iranian vessels in operations.", 1),
    ("Evidence supports that no Iranian vessels were engaged.", 1),
    ("There are no records of Iranian vessels being involved.", 1),
    ("Compliance documents affirm no use of Iranian vessels.", 1),
    ("Proof confirms the non-involvement of Iranian vessels.", 1),
    ("Reports show no involvement of North Korean vessels.", 1),
    ("Documents verify no North Korean vessels were used.", 1),
    ("Certification indicates no North Korean vessel involvement.", 1),
    ("Evidence shows no North Korean vessels engaged.", 1),
    ("Verification reports show no North Korean vessel usage.", 1),
    ("Analysis confirms no North Korean vessels in the fleet.", 1),
    ("Validation shows absence of North Korean vessels.", 1),
    ("Inspections reveal no North Korean vessel participation.", 1)
]

# Function to extract features from text
def extract_features(document):
    words = set(word_tokenize(document))
    features = {}
    for word in words:
        features[word] = (word in words)
    return features

# Generate labeled data
labeled_data = [(extract_features(text), label) for text, label in data]

# Shuffle the labeled data
#random.shuffle(labeled_data)

# Split data into training and testing sets
train_size = int(0.8 * len(labeled_data))
train_set, test_set = labeled_data[:train_size], labeled_data[train_size:]

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

# Classify each sentence
for sentence in sentences:
    features = extract_features(sentence)
    classification = classifier.classify(features)
    print("Sentence:", sentence)
    print("Classification:", classification)
    print()


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Initialize a list to store the predicted labels
predicted_labels = []

# Initialize a list to store the actual labels
actual_labels = []

# Classify each instance in the test set and collect predicted and actual labels
for features, label in test_set:
    actual_labels.append(label)
    predicted_labels.append(classifier.classify(features))

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print("Accuracy:", accuracy)

# Generate classification report
print("Classification Report:")
print(classification_report(actual_labels, predicted_labels))


