import tensorflow as tf
import os
from src.utils import plot_training_history


def train_and_evaluate(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    dataset_name,
    model_name,
    epochs=10,
    batch_size=32,
):
    """
    Обучите и оцените модель.

     Аргументы:
         model: модель Keras для обучения
         X_train, y_train: Обучающие данные и метки
         X_test, y_test: Тестовые данные и метки
         dataset_name (str): Название набора данных
         model_name (str): Название модели ('mlp' или 'cnn')
         epochs (int): Количество периодов обучения
         batch_size (int): Размер пакета

     Результат:
         dict: История тренировки
    """
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy for {model_name} on {dataset_name}: {test_accuracy:.4f}")

    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{dataset_name}_{model_name}.h5")

    # Plot training history
    plot_training_history(
        history.history, dataset_name, model_name
    )  # Исправлено: history.history

    return history.history
