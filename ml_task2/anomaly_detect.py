import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt


from utils.load_data import load_data
from model import RAEAnomaly

def detect_anomaly():
    data = load_data('AAPL')
    dataset = data['log_return']
    # batchy = dataset[:30]
    train_dataset = keras.utils.timeseries_dataset_from_array(
            data=dataset,
            targets=dataset,
            sampling_rate=1,
            sequence_length=60,
            shuffle=True,
            batch_size=256,
            start_index=0,
            end_index=1158
        )
    train_dataset = train_dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)))
    # print(batchy.describe())
    model = RAEAnomaly(60, 1)
    model.load_model()
    result = model.predict(train_dataset)
    # plt.plot(train_dataset, label='Real', marker='o', linestyle='-')
    plt.plot(result[0])
    plt.show()
    # threshold = np.percentile(train_errors, 95)            # определённый порог
    # plot_errors_over_time(dates_test, test_errors, threshold)
    print(result.shape)
    

    # print(result)


detect_anomaly()
