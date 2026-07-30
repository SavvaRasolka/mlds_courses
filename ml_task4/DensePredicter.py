import numpy as np
import json
from dataset.DataSet import DataSet
from models.DenseChatModel import DenseChatModel


class DensePredicter:
    def __init__(self):
        self.dataset = DataSet()
        self.dataset.vectorization()
        self.dataset.index_tags()
        self.model = DenseChatModel(self.dataset.max_tokens)

    def load_model(self):
        self.model.model.load_weights('train/1024densecheckpoint.keras')

    def train(self):
        train_data, val_data, y_train, y_val = self.dataset.split_data()
        self.model.train(train_data, y_train, (val_data, y_val))
        result = self.model.model.evaluate(self.dataset.val_texts, self.dataset.val_tags)
        print("test loss, test acc:", result)

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

    def evaluate(self, dataset):
        _, test_texts, test_tags = self.load_intents(dataset)

        test_labels = np.array([self.dataset.tag_to_idx[tag] for tag in test_tags])
        X_test = self.dataset.vectorizer(test_texts).numpy()
        result = self.model.model.evaluate(X_test, test_labels)
        print(result)

    def predict(self, text):
        prediction = self.model.model.predict(self.dataset.vectorizer([text]).numpy())
        predicted_index = np.argmax(prediction, axis=1)[0]
        print(prediction[0][predicted_index])
        sorted_indicies = sorted(prediction, reverse=True)
        margin = sorted_indicies[0][0] - sorted_indicies[0][1]
        print('margin - ', margin)
        if prediction[0][predicted_index]>0.7 or margin>0.3:
            print(self.dataset.idx_to_tag.get(predicted_index, None))
        else:
            print("не понял")
