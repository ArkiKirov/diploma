from src.data_preprocessing import load_and_preprocess_data
from src.models import build_mlp, build_cnn
from src.train import train_and_evaluate


def run_experiments():
    """
    Run experiments for MLP and CNN on MNIST, Fashion-MNIST, and CIFAR-10.
    """
    datasets = ["mnist", "fashion_mnist", "cifar10"]
    epochs = {"mnist": 5, "fashion_mnist": 20, "cifar10": 30}

    for dataset in datasets:
        print(f"\nRunning experiments for {dataset}...")

        # Load and preprocess data
        X_train, y_train, X_test, y_test = load_and_preprocess_data(dataset)
        input_shape = X_train.shape[1:]

        # Build and train MLP
        mlp_model = build_mlp(input_shape)
        mlp_history = train_and_evaluate(
            mlp_model,
            X_train,
            y_train,
            X_test,
            y_test,
            dataset,
            "mlp",
            epochs=epochs[dataset],
            batch_size=32,
        )

        # Build and train CNN
        cnn_model = build_cnn(input_shape)
        cnn_history = train_and_evaluate(
            cnn_model,
            X_train,
            y_train,
            X_test,
            y_test,
            dataset,
            "cnn",
            epochs=epochs[dataset],
            batch_size=64,
        )


if __name__ == "__main__":
    run_experiments()
