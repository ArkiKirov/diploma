import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_sample_images(X, y, dataset_name, class_names=None):
    """
    Отобразить и сохранить образцы изображений из набора данных.

    Аргументы:
        X: Входные данные (изображения)
        y: Метки (однократные или целые)
        имя_набора данных (str): Название набора данных
        class_names (список): Необязательный список имен классов для обозначения`
    """
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[i], cmap="gray" if dataset_name != "cifar10" else None)
        if class_names:
            plt.title(class_names[np.argmax(y[i])])
        else:
            plt.title(np.argmax(y[i]))
        plt.axis("off")
    plt.suptitle(f"Примеры изображений из {dataset_name}")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{dataset_name}_samples.png")
    plt.close()


def main():
    """
    Загрузить наборы данных и визуализировать образцы изображений для предварительного анализа данных.
    """
    datasets = {
        "mnist": tf.keras.datasets.mnist,
        "fashion_mnist": tf.keras.datasets.fashion_mnist,
        "cifar10": tf.keras.datasets.cifar10,
    }

    class_names = {
        "mnist": [str(i) for i in range(10)],
        "fashion_mnist": [
            "T-shirt",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
        "cifar10": [
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck",
        ],
    }

    for name, dataset in datasets.items():
        (X_train, y_train), (_, _) = dataset.load_data()
        X_train = X_train.astype("float32") / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        plot_sample_images(X_train, y_train, name, class_names[name])


if __name__ == "__main__":
    main()
