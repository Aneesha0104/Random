import numpy as np
import pandas as pd
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from datasets import Dataset, load_metric

# Load data
data = pd.read_csv("augmented_data2_og.csv", encoding='ISO-8859-1')
with open("doc2", 'r') as file:
    test_data = file.read()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
data = data.dropna(subset=['text', 'label'])

# Function to extract countries using spaCy
def extract_countries(text):
    return [ent.text for ent in nlp(text).ents if ent.label_ == 'GPE']

# Preprocess text data
texts, labels = data['text'].values, data['label'].values

# Tokenize data using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=100)
dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': labels
})

# Split dataset into training and testing sets
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2, seed=42).values()

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    logging_first_step=True,  # To log the first step
    log_level='error'  # Suppress unnecessary logging
)

# Define metric for evaluation
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = (torch.sigmoid(torch.tensor(pred.predictions)) > 0.5).int().numpy().flatten()
    acc = accuracy_metric.compute(predictions=preds, references=labels)['accuracy']
    prec = precision_metric.compute(predictions=preds, references=labels)['precision']
    rec = recall_metric.compute(predictions=preds, references=labels)['recall']
    f1 = f1_metric.compute(predictions=preds, references=labels)['f1']
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Metrics:")
for key, value in results.items():
    if key.startswith("eval_"):
        print(f"{key[5:].capitalize()}: {value:.4f}")

import pickle

# Define the path where you want to save the model
model_path_pickle = "bert_model.pkl"

# Save the model
with open(model_path_pickle, 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully at:", model_path_pickle)

# Function to classify and extract countries
def classify_and_extract(model, sentences):
    results = set()
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=100, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings)
    predictions = torch.sigmoid(outputs.logits).numpy().flatten()
    for sentence, prediction in zip(sentences, predictions):
        if prediction > 0.5:
            countries = extract_countries(sentence)
            if countries:
                results.add((sentence, tuple(countries)))
    return results

# Tokenize test data into sentences
sentences = sent_tokenize(test_data)


print("BERT Model Exclusion Sentences with Countries:")
for sentence, countries in classify_and_extract(model, sentences):
    print("Sentence:", sentence)
    print("Countries:", countries)
    print()
