import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


from utils.load_data import load_data
from model import RAEAnomaly

def train_model():
    
    data = load_data('GOOGL')
    dataset = data[['log_return']]

    # scaler = StandardScaler()
    # dataset = scaler.fit_transform(raw_dataset)

    num_train_samples = 1158
    num_val_samples = 579
    sampling_rate = 1
    sequence_lenght = 60
    batch_size = 256
    train_dataset = keras.utils.timeseries_dataset_from_array(
        data=dataset,
        targets=dataset,
        sampling_rate=sampling_rate,
        sequence_length=sequence_lenght,
        shuffle=True,
        batch_size=batch_size,
        start_index=0,
        end_index=num_train_samples
    )
    train_dataset = train_dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)))
    val_dataset = keras.utils.timeseries_dataset_from_array(
            data=dataset,
            targets=dataset,
            sampling_rate=sampling_rate,
            sequence_length=sequence_lenght,
            shuffle=True,
            batch_size=batch_size,
            start_index=num_train_samples,
            end_index=num_train_samples + num_val_samples
        )
    val_dataset = val_dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)))
    test_dataset = keras.utils.timeseries_dataset_from_array(
            data=dataset,
            targets=dataset,
            sampling_rate=sampling_rate,
            sequence_length=sequence_lenght,
            shuffle=True,
            batch_size=batch_size,
            start_index=num_train_samples + num_val_samples
        )
    test_dataset = test_dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)))
    model = RAEAnomaly(seq_length=sequence_lenght, data_shape=1)
    model.train(train_data=train_dataset, val_data=val_dataset)
    print(model.evaluate(test_dataset))
    
train_model()
