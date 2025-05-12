import tensorflow as tf


def build_mlp(input_shape):
    """
    Создать многослойную модель персептрона.

    Аргументы:
        input_shape (кортеж): Форма входных данных (например, (28, 28, 1) для MNIST)

    Результат:
        tf.keras.Модель: модель MLP
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
    Создать сверточную модель нейронной сети.

    Аргументы:
        input_shape (кортеж): форма входных данных (например, (32, 32, 3) для CIFAR-10)

    Результат:
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
