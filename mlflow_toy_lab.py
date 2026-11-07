import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.keras

def run_mlflow_toy_lab():
    """
    Sub-task 3.1: MLflow Locally Tool Lab
    Trains two runs of a tiny Keras model with different learning rates (0.01 vs 0.001).
    Logs parameters, per-epoch metrics, loss/accuracy plots, and model artifacts.
    """
    print("\n" + "="*80)
    print("SUB-TASK 3.1: MLFLOW LOCAL TOOL LAB")
    print("="*80)

    # Use SQLite database backend for reliable local tracking & Model Registry
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "Toy_Experiment_Lab_23L_3091"
    mlflow.set_experiment(experiment_name)

    # Simple synthetic dataset for instantaneous execution
    np.random.seed(42)
    x_train = np.random.randn(200, 10).astype(np.float32)
    y_train = (np.sum(x_train, axis=1) > 0).astype(np.float32)
    x_val = np.random.randn(50, 10).astype(np.float32)
    y_val = (np.sum(x_val, axis=1) > 0).astype(np.float32)

    def build_toy_model():
        model = keras.Sequential([
            layers.Dense(16, activation="relu", input_shape=(10,)),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        return model

    learning_rates = [0.01, 0.001]

    for idx, lr in enumerate(learning_rates, 1):
        run_name = f"toy_run_lr_{lr}"
        print(f"\n[*] Starting MLflow Run {idx}/2: {run_name} (learning_rate={lr})...")
        
        with mlflow.start_run(run_name=run_name):
            # 1. Log Parameters
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", 16)
            mlflow.log_param("epochs", 10)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("architecture", "Dense(16)-Dense(8)-Dense(1)")

            # Set Tags
            mlflow.set_tag("student_id", "23L-3091")
            mlflow.set_tag("task", "3.1_MLflow_Locally")
            mlflow.set_tag("stage", "development")

            # 2. Build and Train Model
            model = build_toy_model()
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            # Custom logging callback for per-epoch metrics
            class MLflowEpochCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        for metric_name, val in logs.items():
                            mlflow.log_metric(metric_name, float(val), step=epoch)

            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=10,
                batch_size=16,
                callbacks=[MLflowEpochCallback()],
                verbose=0
            )

            # 3. Log Final Metrics
            final_val_loss = history.history["val_loss"][-1]
            final_val_acc = history.history["val_accuracy"][-1]
            mlflow.log_metric("final_val_loss", float(final_val_loss))
            mlflow.log_metric("final_val_accuracy", float(final_val_acc))

            # 4. Create and Log Plot Artifact
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history["loss"], label="Train Loss", color="red")
            plt.plot(history.history["val_loss"], label="Val Loss", color="blue")
            plt.title(f"Loss (lr={lr})")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history["accuracy"], label="Train Acc", color="green")
            plt.plot(history.history["val_accuracy"], label="Val Acc", color="purple")
            plt.title(f"Accuracy (lr={lr})")
            plt.legend()
            plt.tight_layout()

            plot_path = f"toy_curves_lr_{lr}.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path="plots")

            # 5. Log Model Artifact
            try:
                mlflow.tensorflow.log_model(model, artifact_path="model")
            except Exception:
                model_path = f"toy_model_lr_{lr}.keras"
                model.save(model_path)
                mlflow.log_artifact(model_path, artifact_path="model")

            print(f"[+] Run {run_name} completed and logged to MLflow:")
            print(f"    - final_val_loss    : {final_val_loss:.4f}")
            print(f"    - final_val_accuracy: {final_val_acc:.4f}")

    print("\n" + "="*80)
    print("[SUCCESS] Both toy runs have been logged to local MLflow (sqlite:///mlflow.db)!")
    print("To view and compare runs in the MLflow UI, run:")
    print("  .\\.venv\\Scripts\\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")
    print("Then open: http://localhost:5001 in your browser.")
    print("="*80)

if __name__ == "__main__":
    run_mlflow_toy_lab()
