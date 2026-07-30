import json
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split


class DataSet:
    def __init__(self):
        self.max_tokens = 1300
        self.train_texts, self.train_tags = self.load_intents('train_intents.json')
        self.val_texts, self.val_tags = self.load_intents('val_intents.json')
        self.unique_tags = sorted(set(self.train_tags))
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.unique_tags)}
        self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}
        self.num_classes = len(self.unique_tags) 
        self.vectorizer = layers.TextVectorization(
                                max_tokens=self.max_tokens,  
                                ngrams=2,              
                                standardize='lower', 
                                output_mode='tf_idf'                 
                               )


    def index_tags(self):
        self.train_tags = self.index_label(self.train_tags)
        self.val_tags = self.index_label(self.val_tags)

    def index_label(self, labels):
        return np.array([self.tag_to_idx[tag] for tag in labels])

    def load_intents(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts, tags = [], []
        for intent in data['intents']:
            tag = intent['tag']
            for pattern in intent['patterns']:
                texts.append(pattern)
                tags.append(tag)
        return texts, tags

    def vectorization(self):
        self.vectorizer.adapt(self.train_texts)
        self.train_texts = self.vectorizer(self.train_texts).numpy()
        self.val_texts = self.vectorizer(self.val_texts).numpy()

    def split_data(self):
        train_data, val_data, y_train, y_val = train_test_split(
                                                self.train_texts, self.train_tags,
                                                  test_size=0.2, shuffle=True, random_state=42
                                                )
        return train_data, val_data, y_train, y_val
       