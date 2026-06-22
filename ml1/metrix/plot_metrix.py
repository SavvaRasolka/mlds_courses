import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def show_metrix(target, pred):
    accuracy = accuracy_score(target, pred)
    print(f"Точность модели: {accuracy:.4f}")
    print("Метрики качества:")
    print(classification_report(target, pred))


def show_confusion_matrix(target, pred, classes):
    matrix = confusion_matrix(target, pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Предсказанные классы')
    plt.ylabel('Истинные классы')
    plt.title('Матрица ошибок')
    plt.show()
