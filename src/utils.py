import matplotlib.pyplot as plt
import os


def plot_training_history(history, dataset_name, model_name):
    """
    Обучение пилота и проверка точности/потери.

    Аргументы:
        история: История обучения Keras
        dataset_name (str): Название набора данных
        model_name (str): Название модели.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history["accuracy"], label="Train Accuracy")
    ax1.plot(history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title(f"{model_name} Accuracy on {dataset_name}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history["loss"], label="Train Loss")
    ax2.plot(history["val_loss"], label="Validation Loss")
    ax2.set_title(f"{model_name} Loss on {dataset_name}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs("notebooks/plots", exist_ok=True)
    plt.savefig(f"notebooks/plots/{dataset_name}_{model_name}_history.png")
    plt.close()
