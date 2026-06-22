from sklearn.ensemble import AdaBoostClassifier


def predict_AdaBoost(train_data, train_target, test_data):
    adabc = AdaBoostClassifier(random_state=42)
    model = adabc.fit(train_data, train_target)
    prediction_target = model.predict(test_data)
    return prediction_target, model
