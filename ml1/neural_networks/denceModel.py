from tensorflow import keras
from tensorflow.keras import layers


class DenceClassificator:

    def __init__(self):
        self.model = keras.Sequential([
            layers.Dense(4096, activation="relu"),
            layers.Dense(512, activation='relu'),
            layers.Dense(10, activation="softmax")
        ])
        self.model.compile(optimizer='adam',
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
        self.model.summary()

    def train(self, train_data, targets):
        history = self.model.fit(train_data, targets, epochs=400, batch_size=512, validation_split=0.2)
        return history
