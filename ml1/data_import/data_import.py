import cv2
import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


NUM_TRAIN_IMAGES = 50000
NUM_TEST_IMAGES = 10000
COLOUR_DEPTH = 255


class cifarDataset:
    
    def __init__(self):
        self.images_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

    def convert_to_grayscale(self, images):
        grayscale_images = np.zeros((images.shape[0], 32, 32, 1))
        
        for i in range(images.shape[0]):
            grayscale_images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY).reshape(32, 32, 1)
        
        return grayscale_images

    def flat_data(self):
        self.y_train = self.y_train.ravel()
        self.y_test = self.y_test.ravel()
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_train = self.x_train.astype('float32')/COLOUR_DEPTH
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)
        self.x_test = self.x_test.astype('float32')/COLOUR_DEPTH

    def cnn_prepare(self):
        self.x_train = self.x_train.reshape((NUM_TRAIN_IMAGES, 32, 32, 3))
        self.x_train = self.x_train.astype('float32')/COLOUR_DEPTH
        self.x_test = self.x_test.reshape((NUM_TEST_IMAGES, 32, 32, 3))
        self.x_test = self.x_test.astype('float32')/COLOUR_DEPTH

    def draw_classes(self):
        classes, counts = np.unique(self.y_train, return_counts=True)
        plt.bar(self.images_classes, counts)
        plt.show()
