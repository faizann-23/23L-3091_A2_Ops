import keras
import mlflow
import numpy as np
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow_local.db")
mlflow.set_experiment("autolog_check")
try:
    mlflow.tensorflow.autolog(log_datasets=False, checkpoint=False)
    print("autolog accepted checkpoint=False")
except TypeError:
    mlflow.tensorflow.autolog(log_datasets=False)
    print("autolog: checkpoint option not available")

x = np.random.rand(256, 8).astype("float32")
y = (x.sum(1) > 4).astype("float32")
model = keras.Sequential([keras.Input(shape=(8,)), keras.layers.Dense(8, activation="relu"),
                          keras.layers.Dense(1, activation="sigmoid")])
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
with mlflow.start_run() as run:
    model.fit(x, y, validation_split=0.2, epochs=3, batch_size=32, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)])
client = MlflowClient()
data = client.get_run(run.info.run_id).data
print("params  :", sorted(data.params))
print("metrics :", sorted(data.metrics))
print("artifacts:", [f.path for f in client.list_artifacts(run.info.run_id)])