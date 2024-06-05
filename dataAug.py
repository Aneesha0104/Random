import random
import pandas as pd
import numpy as np  # Add this import for handling NaN values
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

# Data
data = pd.read_csv('augmented_data2_3.csv')


# Remove rows with NaN values
data.dropna(inplace=True)

# Parameters
TOPK = 20  # default=100
NUM_AUGMENTATIONS = 4
NUM_DATA_POINTS = 100

# Augmentation models
aug_bert_insert = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action='insert', top_k=TOPK)
#aug_bert_substitute = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action='substitute', top_k=TOPK)
aug_synonym = naw.SynonymAug(aug_src='wordnet')
#aug_random_swap = naw.RandomWordAug(action='swap')
aug_random_delete = naw.RandomWordAug(action='delete')

# List of word-level augmenters
word_augmenters = [aug_bert_insert, aug_synonym, aug_random_delete]

# Sentence-level augmenter
sentence_paraphraser = nas.ContextualWordEmbsForSentenceAug(model_path='GPT2')

# Augmented data
augmented_data = []

for _ in range(NUM_DATA_POINTS):
    for _, row in data.iterrows():
        sentence = row['sentence']
        label = row['label']

        augmented_data.append((sentence, label))  # Original sentence
        for _ in range(NUM_AUGMENTATIONS):
            augmenter = random.choice(word_augmenters)
            augmented_sentence = augmenter.augment(sentence)
            augmented_data.append((augmented_sentence, label))
        # Sentence-level augmentation
        augmented_sentence = sentence_paraphraser.augment(sentence)
        augmented_data.append((augmented_sentence, label))

# Shuffle the augmented data
random.shuffle(augmented_data)

# Convert to DataFrame
df = pd.DataFrame(augmented_data, columns=['sentence', 'label'])

# Save to CSV
df.to_csv('augmented_data2_5.csv', index=False)
