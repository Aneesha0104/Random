import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the document to be tested
file_path = "doc1"
with open(file_path, 'r') as file:
    test_data = file.read()

# Training data
train_data = [
    ("All Documents provided to evidence no Iran vessel involved", 1),
    ("Documentation confirming no Iran vessel involved in trading.", 1),
    ("Proof of no Iran vessel involvement through submitted documents", 1),
    ("This is a random statement.", 0),
    ("Another non-exclusion statement.", 0),
    ("This is definitely not about vessels.", 0),


]

# Naive Bayes Model Training
texts, labels = zip(*train_data)

x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(x_train, y_train)

accuracy = pipeline.score(x_test, y_test)
print(f"Accuracy of NB model: {accuracy}")

# Function to identify exclusion statements and extract the country
def identify_exclusion_statements(document):
    lines = document.split("\n")
    exclusion_statements = []
    countries = []
    for line in lines:
        if pipeline.predict([line]) == 1:
            exclusion_statements.append(line)
            country_match = re.search(r'\bno (\w+)', line, re.IGNORECASE)
            if country_match:
                countries.append(country_match.group(1))
    return exclusion_statements, countries

# Identify exclusion statements and extract countries from the test document
exclusion_statements, countries = identify_exclusion_statements(test_data)
print("Exclusion Statements:")
for statement in exclusion_statements:
    print(statement)
print("\nCountries:")
for country in countries:
    print(country)
