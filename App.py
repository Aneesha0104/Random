import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grids
param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
param_grid_dt = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30, 40, 50]}
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}
param_grid_gbm = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]}
param_grid_xgb = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]}
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
param_grid_nb = {'alpha': [0.01, 0.1, 1, 10]}

# Initialize classifiers with hyperparameter tuning
classifiers = {
    # 'Support Vector Machine': GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=2),
    # 'Decision Tree': GridSearchCV(DecisionTreeClassifier(), param_grid_dt, refit=True, verbose=2),
    'Random Forest': GridSearchCV(RandomForestClassifier(), param_grid_rf, refit=True, verbose=2),
    # 'Gradient Boosting Machines': GridSearchCV(GradientBoostingClassifier(), param_grid_gbm, refit=True, verbose=2),
    # 'XGBoost': GridSearchCV(xgb.XGBClassifier(), param_grid_xgb, refit=True, verbose=2),
    # 'CatBoost': cb.CatBoostClassifier(),  # CatBoost has its own internal hyperparameter tuning mechanism
    # 'K-Nearest Neighbors': GridSearchCV(KNeighborsClassifier(), param_grid_knn, refit=True, verbose=2),
    'Naive Bayes': GridSearchCV(MultinomialNB(), param_grid_nb, refit=True, verbose=2)
}

# Function to evaluate classifier
def evaluate_classifier(name, classifier, X_train, y_train, X_test, y_test, test_data):
    classifier.fit(X_train, y_train)
    if name != 'CatBoost':
        best_params = classifier.best_params_
        classifier = classifier.best_estimator_
        # st.write(f"Best Parameters for {name}: {best_params}")

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

    return accuracy, exclusion_with_countries

# Streamlit App
st.title('Text Classification and Country Extraction')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    data = data.dropna(subset=['text', 'label'])
    X = vectorizer.fit_transform(data['text'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("Select a classifier from the list below:")
classifier_name = st.selectbox('Select Classifier', list(classifiers.keys()))

test_text = st.text_area('Enter text to classify and extract countries:', '')

if st.button('Classify and Extract'):
    if test_text and classifier_name:
        classifier = classifiers[classifier_name]
        accuracy, exclusion_with_countries = evaluate_classifier(
            classifier_name, classifier, X_train, y_train, X_test, y_test, test_text
        )

        st.write(f"**{classifier_name}**")
        st.write(f"Test set accuracy: {accuracy}")

        st.write("**Sentences labeled as exclusion along with extracted countries:**")
        if exclusion_with_countries:
            for sentence, countries in exclusion_with_countries:
                st.write(f"**Sentence:** {sentence}")
                st.write(f"**Countries:** {', '.join(countries)}")
                st.write("---")
        else:
            st.write("No sentences were classified as exclusion or no countries were found.")
    else:
        st.write("Please enter the text and select a classifier.")

