from sklearn.svm import SVC


def create_SVC(train_data, train_target, test_data):
    svc = SVC(kernel='rbf', C=0.5, gamma='scale')
    model = svc.fit(train_data, train_target)
    prediction_target = model.predict(test_data)
    return prediction_target, model
