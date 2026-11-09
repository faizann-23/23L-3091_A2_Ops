"""3.1 / 3.2 toy lab: log two runs with different learning rates (local or DagsHub)."""
import sys

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras

from tracking import setup_tracking

target = sys.argv[1] if len(sys.argv) > 1 else "local"   # "local" or "dagshub"
setup_tracking("toy_lab", target)

(x, y), _ = keras.datasets.fashion_mnist.load_data()
x = x.astype("float32") / 255.0
x_train, y_train = x[:10000], y[:10000]
x_val, y_val = x[10000:12000], y[10000:12000]


class LogEpochs(keras.callbacks.Callback):
    """Send every per-epoch metric of Keras to MLflow."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


for lr in (1e-2, 1e-4):
    keras.utils.set_random_seed(42)
    with mlflow.start_run(run_name=f"toy_lr_{lr:g}"):
        mlflow.log_params({"learning_rate": lr, "units": 64, "epochs": 5, "batch_size": 64})
        mlflow.set_tags({"purpose": "mlflow_lab", "dataset": "fashion_mnist_subset", "target": target})

        model = keras.Sequential([
            keras.Input(shape=(28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                         epochs=5, batch_size=64, callbacks=[LogEpochs()], verbose=2)

        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(hist.history["loss"], label="train loss")
        ax.plot(hist.history["val_loss"], label="validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Toy model, learning rate = {lr:g}")
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "loss_curve.png")   # one plot as an artifact
        plt.close(fig)

        mlflow.log_metric("final_val_accuracy", float(hist.history["val_accuracy"][-1]))
        try:
            mlflow.keras.log_model(model, name="model")           # MLflow 3.x
        except TypeError:
            mlflow.keras.log_model(model, artifact_path="model")  # older MLflow
        print(f"run finished: lr={lr:g} | final val_accuracy={hist.history['val_accuracy'][-1]:.4f}")