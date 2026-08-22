from tensorflow.keras import layers
from datetime import datetime
from tensorflow import keras
import numpy as np 


class RAEAnomaly:
    def __init__(self, seq_length, data_shape):
        inputs = keras.Input(shape=(seq_length, data_shape))

        x = layers.LSTM(256, activation='tanh', recurrent_dropout=0.2, return_sequences=True)(inputs)

        x = layers.LSTM(128, activation='tanh', recurrent_dropout=0.2)(x)

        x = layers.Dropout(0.5)(x)
        
        x = layers.RepeatVector(seq_length)(x)

        x = layers.LSTM(128, activation='tanh', recurrent_dropout=0.2,  return_sequences=True)(x)
        
        x = layers.LSTM(256, activation='tanh', return_sequences=True)(x)

        outputs = layers.TimeDistributed(layers.Dense(data_shape))(x)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam',
                           loss="mse",
                           metrics=["mae"])
        self.model.summary()

    def load_model(self):
        self.model.load_weights('train/checkpoint.keras')

    def predict(self, data):
        pred = self.model.predict(data)
        return pred
        # return np.mean(np.square(data - pred), axis=(1, 2))

    def train(self, train_data, val_data):
        train_dir = "trainDense"  + datetime.now().strftime("%Y%m%d-%H%M%S")
        history = self.model.fit(train_data, epochs=100, validation_data=val_data, batch_size=256,
                                callbacks=[ 
                                    keras.callbacks.TensorBoard(log_dir=train_dir),
                                    keras.callbacks.EarlyStopping(monitor='mae', patience=5),
                                    keras.callbacks.ModelCheckpoint(filepath="train/checkpoint.keras",
                                                                    monitor='val_loss',
                                                                    save_best_only=True
                                                                    )
                                ]
                                )
        return history

    def evaluate(self, data):
        return self.model.evaluate(data)
    