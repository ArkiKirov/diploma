import tensorflow as tf


def build_mlp(input_shape):
    """
    Build a Multi-Layer Perceptron model.

    Args:
        input_shape (tuple): Shape of input data (e.g., (28, 28, 1) for MNIST)

    Returns:
        tf.keras.Model: MLP model
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def build_cnn(input_shape):
    """
    Build a Convolutional Neural Network model.

    Args:
        input_shape (tuple): Shape of input data (e.g., (32, 32, 3) for CIFAR-10)

    Returns:
        tf.keras.Model: CNN model
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model
