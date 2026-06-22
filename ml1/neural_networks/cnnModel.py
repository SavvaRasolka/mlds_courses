from tensorflow import keras
from tensorflow.keras import layers


class CNNClassificator:

    def __init__(self):
        inputs = keras.Input(shape=(32, 32, 3))
        x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='valid')(inputs)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='valid')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='valid')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1536, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        self.model = keras.Model(inputs = inputs, outputs = outputs)
        self.model.compile(optimizer='adam',
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
        self.model.summary()
    
    def train(self, train_data, targets):
        history = self.model.fit(train_data, targets, epochs=130, batch_size=256)
        return history
            