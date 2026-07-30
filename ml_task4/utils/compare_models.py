from Predicter import Predicter
from DensePredicter import DensePredicter


def compare_models():
    dense = DensePredicter()
    dense.load_model()
    dense.evaluate('dataset/evaluate.json')
    lstm = Predicter()
    lstm.load_model()
    lstm.evaluate('dataset/evaluate.json')
