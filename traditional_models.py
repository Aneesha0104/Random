import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import spacy

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
data = data.dropna(subset=['text'])
data = data.dropna(subset=['label'])
# Use CountVectorizer to transform the text data into feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split data into training and testing sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting Machines': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    # 'LightGBM': lgb.LGBMClassifier(),
    'CatBoost': cb.CatBoostClassifier(verbose=0),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': MultinomialNB()
}

# Function to evaluate and print results for each classifier
def evaluate_classifier(name, classifier, X_train, y_train, X_test, y_test):
    print(f"Classifier: {name}")
    classifier.fit(X_train, y_train)
    cv_scores = cross_val_score(classifier, X, y, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", cv_scores.mean())
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

    # Generate classification report
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    print("\n" + "="*80 + "\n")

# Evaluate each classifier
for name, classifier in classifiers.items():
    evaluate_classifier(name, classifier, X_train, y_train, X_test, y_test)
