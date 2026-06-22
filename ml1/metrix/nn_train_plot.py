import matplotlib.pyplot as plt


def draw_quality_plot(epoch, loss):
    plt.plot(epoch, loss, "-b")
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.show()

def draw_accuracy_plot(epoch, accuracy):
    plt.plot(epoch, accuracy, "-b")
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()
    