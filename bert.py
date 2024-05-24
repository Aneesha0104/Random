import numpy as np
import pandas as pd
import spacy
import torch
from transformers import BertTokenizer,BertForSequenceClassification,Trainer,TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import sent_tokenize
from datasets import Dataset,load_metric

data = pd.read_csv("augmented_data2_og.csv", encoding='ISO-8859-1')
with open("doc2", 'r') as file:
    test_data = file.read()
nlp = spacy.load("en_core_web_sm")
data = data.dropna(subset=['text', 'label'])

def extract_countries(text):
    return [ent.text for ent in nlp(text).ents if ent.label_ == 'GPE']

texts, labels = data['text'].values, data['label'].values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=100)
dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': labels
})
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2, seed=42).values()

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()


results = trainer.evaluate()
print("Test set accuracy:", results['eval_accuracy'])


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


sentences = sent_tokenize(test_data)


print("BERT Model Exclusion Sentences with Countries:")
for sentence, countries in classify_and_extract(model, sentences):
    print("Sentence:", sentence)
    print("Countries:", countries)
    print()
