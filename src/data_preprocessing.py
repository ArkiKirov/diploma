import tensorflow as tf
import numpy as np


def load_and_preprocess_data(dataset_name):
    """
    Load and preprocess dataset (MNIST, Fashion-MNIST, or CIFAR-10).

    Args:
        dataset_name (str): Name of the dataset ('mnist', 'fashion_mnist', 'cifar10')

    Returns:
        tuple: (X_train, y_train, X_test, y_test) preprocessed data
    """
    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )
    elif dataset_name == "cifar10":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError(
            "Unknown dataset: choose 'mnist', 'fashion_mnist', or 'cifar10'"
        )

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Reshape for MLP (flatten images)
    if dataset_name in ["mnist", "fashion_mnist"]:
        X_train = X_train.reshape(-1, 28, 28, 1)  # Add channel dimension for CNN
        X_test = X_test.reshape(-1, 28, 28, 1)
    elif dataset_name == "cifar10":
        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test
