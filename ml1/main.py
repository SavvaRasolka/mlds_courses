from data_import.data_import import cifarDataset
from neural_networks.denceModel import DenceClassificator
from neural_networks.cnnModel import CNNClassificator
from models.svc import create_SVC
from models.RFxgboost import predict_XGBoost
from models.randomForest import predict_randomforest
from metrix.nn_train_plot import draw_quality_plot, draw_accuracy_plot
from metrix.plot_metrix import show_confusion_matrix, show_metrix
from metrix.timer_wraapper import timer_wrapper
import matplotlib.pyplot as plt


def main():
    mydataset = cifarDataset()
    mydataset.draw_classes()
    classificator = CNNClassificator()

    train_history = classificator.train(mydataset.x_train, mydataset.y_train)
    draw_quality_plot( train_history.epoch, train_history.history['loss'])
    draw_accuracy_plot( train_history.epoch, train_history.history['accuracy'])
    
    prediction = classificator.model.predict(mydataset.x_train, mydataset.y_train, mydataset.x_test)
    show_metrix(mydataset.y_test, prediction)
    show_confusion_matrix(mydataset.y_test, prediction, mydataset.images_classes)


if __name__ == '__main__':
    main()
