# import random
# import pandas as pd
# import nlpaug.augmenter.word as naw
#
# # Data
# data = [
#     ("All Documents provided to evidence no Iran vessel involved", 1),
#     ("Documentation confirming no Iran vessel involved in trading.", 1),
#     ("Proof of no Iran vessel involvement through submitted documents", 1),
#     ("This is a random statement.", 0),
#     ("Another non-exclusion statement.", 0),
#     ("This is definitely not about vessels.", 0),
#     ("Records showing no Syrian vessel participation.", 1),
#     ("Verification documents ensure no North Korean vessel engaged.", 1),
#     ("Confirmed: no vessels from Cuba were used.", 1),
#     ("No evidence of any Libyan vessel involvement found.", 1),
#     ("Reports confirming absence of Sudanese vessels in the trade.", 1),
#     ("Official documents prove no Russian vessel engagement.", 1),
#     ("Ensuring no Venezuelan vessel was utilized.", 1),
#     ("No Myanmar vessel has been documented in the activities.", 1),
#     ("Unrelated statement without any specific details.", 0),
#     ("A completely irrelevant sentence.", 0),
# ]
#
# # Parameters
# TOPK = 20  # default=100
# ACT = 'insert'  # "substitute"
# NUM_AUGMENTATIONS = 3
# NUM_DATA_POINTS = 200
#
# # Augmentation model
# aug_bert = naw.ContextualWordEmbsAug(
#     model_path='distilbert-base-uncased',
#     action=ACT,
#     top_k=TOPK
# )
#
# # Augmented data
# augmented_data = []
#
# for _ in range(NUM_DATA_POINTS):
#     for sentence, label in data:
#         augmented_data.append((sentence, label))  # Original sentence
#         for _ in range(NUM_AUGMENTATIONS):
#             augmented_sentence = aug_bert.augment(sentence)
#             augmented_data.append((augmented_sentence, label))
#
# # Shuffle the augmented data
# random.shuffle(augmented_data)
#
# # Convert to DataFrame
# df = pd.DataFrame(augmented_data, columns=['sentence', 'label'])
#
# # Save to CSV
# df.to_csv('augmented_data2_1.csv', index=False)
#
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from nltk.tokenize import sent_tokenize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import catboost as cb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Function to extract countries from text using spaCy
def extract_countries(text):
    doc = nlp(text)
    countries = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    return countries


# Load data
file_path = "augmented_data2_og.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1')
data = data.dropna(subset=['text', 'label'])

# Prepare the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting Machines': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'CatBoost': cb.CatBoostClassifier(verbose=0),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': MultinomialNB()
}


# Function to evaluate classifier
def evaluate_classifier(name, classifier, X_train, y_train, X_test, y_test, test_data):
    classifier.fit(X_train, y_train)
    cv_scores = cross_val_score(classifier, X, y, cv=5)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    sentences = sent_tokenize(test_data)
    exclusion_with_countries = []

    for sentence in sentences:
        features = vectorizer.transform([sentence])
        classification = classifier.predict(features)[0]
        if classification == 1:
            countries = extract_countries(sentence)
            if countries:
                exclusion_with_countries.append((sentence, countries))

    return cv_scores, accuracy, exclusion_with_countries

for sentence, countries in  exclusion_with_countries:
    print("Sentence:", sentence)
    print("Countries:", countries)
    print()