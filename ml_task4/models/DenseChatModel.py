from tensorflow.keras import layers
from datetime import datetime
from tensorflow import keras



class DenseChatModel:
    def __init__(self, max_tokens):
        inputs = keras.Input(shape=(max_tokens,))
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(11, activation='softmax')(x)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam',
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
        self.model.summary()

    def train(self, train_data, targets, val):
        train_dir = "trainDense"  + datetime.now().strftime("%Y%m%d-%H%M%S")
        history = self.model.fit(train_data, targets, epochs=100, validation_data=val, batch_size=256, shuffle=True, validation_split=0.2,
                                callbacks=[ 
                                    keras.callbacks.TensorBoard(log_dir=train_dir),
                                    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
                                    keras.callbacks.ModelCheckpoint(filepath="train/checkpoint.keras",
                                                                    monitor='val_loss',
                                                                    save_best_only=True
                                                                    )
                                ]
                                )
        return history
