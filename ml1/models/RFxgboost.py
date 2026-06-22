from xgboost import XGBClassifier


def predict_XGBoost(train_data, train_target, test_data):
    axgbclassifier = XGBClassifier(objective='multi:softmax')
    model = axgbclassifier.fit(train_data, train_target)
    prediction_target = model.predict(test_data)
    return prediction_target, model
