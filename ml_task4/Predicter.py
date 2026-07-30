import json
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split

from models.ChatModel import ChatModel


class Predicter:
    def __init__(self):
        self.vectorizer = layers.TextVectorization(               
                                            standardize='lower',
                                            split='whitespace',                 
                                            output_mode='int'                  
                                            )
        self.data, self.texts, self.tags = self.load_intents('dataset/train_intents.json') 
        self.vectorizer.adapt(self.texts)
        self.model = self.create_embedding_matrix()

        unique_tags = sorted(set(self.tags))
        self.tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
        self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}


    def create_embedding_matrix(self):
        max_tokens = self.vectorizer.vocabulary_size()
        embedding_dim = 300
        embedding_index = {}
        with open('models/ft_native_300_ru_wiki_lenta_lower_case.vec', 'r', encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embedding_index[word] = coefs

        vocabulary = self.vectorizer.get_vocabulary()
        word_index = dict(zip(vocabulary, range(len(vocabulary))))

        embedding_matrix = np.zeros((max_tokens, embedding_dim))
        for word, i in word_index.items():
            if i < max_tokens:
                embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        model = ChatModel(embedding_matrix, max_tokens)
        return model

    def load_model(self):
        self.model.model.load_weights('train/1024lstmcheckpoint.keras')

    def load_intents(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts, tags = [], []
        for intent in data['intents']:
            tag = intent['tag']
            for pattern in intent['patterns']:
                texts.append(pattern)
                tags.append(tag)
        return data, texts, tags

    def split_data(self):
            train_data, val_data, y_train, y_val = train_test_split(
                                                    self.train_texts, self.train_tags,
                                                      test_size=0.2, shuffle=True, random_state=42
                                                    )
            return train_data, val_data, y_train, y_val

    def train(self):

        train_data, val_data, y_train, y_val = self.split_data()
        self.model.train(train_data, y_train, (val_data, y_val))
        # result = self.model.model.evaluate(self.dataset.val_texts, self.dataset.val_tags)
        # print("test loss, test acc:", result)

    def predict_text(self, text:str):
        text = text.lower()
        prediction = self.model.model.predict(self.vectorizer([text]).numpy())
        predicted_index = np.argmax(prediction, axis=1)[0]
        print('max pred - ', prediction[0][predicted_index])
        sorted_indicies = sorted(prediction, reverse=True)
        margin = sorted_indicies[0][0] - sorted_indicies[0][1]
        print('margin - ', margin)
        if prediction[0][predicted_index]>0.7 or margin>0.3:
            answer, tag = self.index_to_answer(predicted_index)
            answer = answer[0]
        else:
            answer = "не понял"
            tag = "uknown"
        return prediction, prediction[0][predicted_index], margin, answer, tag

    def index_to_answer(self, idx):
        tag = self.idx_to_tag.get(idx, None)
        intents = self.data.get("intents", [])
        for intent in intents:
            if intent.get("tag") == tag:
                print(tag)
                return [intent.get("responses", []), tag]

    def evaluate(self, dataset):
        _, test_texts, test_tags = self.load_intents('val_intents.json')
        test_labels = np.array([self.tag_to_idx[tag] for tag in test_tags])
        X_test = self.vectorizer(test_texts).numpy()
        result = self.model.model.evaluate(X_test, test_labels)
        print(result)

