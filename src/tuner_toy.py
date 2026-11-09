"""2.7 Toy part: Keras Tuner random search on Fashion-MNIST (learning rate and units)."""
from pathlib import Path

import keras
import keras_tuner as kt
import pandas as pd

SEED = 42
keras.utils.set_random_seed(SEED)

(x, y), _ = keras.datasets.fashion_mnist.load_data()
x = x.astype("float32") / 255.0
x_train, y_train = x[:10000], y[:10000]   # small subset so the toy search is quick
x_val, y_val = x[10000:12000], y[10000:12000]


def build(hp):
    units = hp.Int("units", min_value=32, max_value=256, step=32)
    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(units, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=8, seed=SEED,
                        directory="tuner_logs", project_name="toy_fashion_mnist", overwrite=True)
tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=5, verbose=0)

rows = []
for t in tuner.oracle.get_best_trials(num_trials=8):
    rows.append({"trial": t.trial_id,
                 "units": t.hyperparameters.get("units"),
                 "learning_rate": round(float(t.hyperparameters.get("learning_rate")), 5),
                 "val_accuracy": round(float(t.score), 4)})
df = pd.DataFrame(rows)
Path("reports").mkdir(exist_ok=True)
df.to_csv("reports/tuner_toy_results.csv", index=False)

best = tuner.get_best_hyperparameters(1)[0]
text = ("TOY TUNING RESULTS (Fashion-MNIST, random search, 8 trials, best first)\n\n"
        + df.to_string(index=False)
        + f"\n\nBest settings: units = {best.get('units')}, "
          f"learning_rate = {best.get('learning_rate'):.5f}")
print(text)
Path("reports/tuner_toy_summary.txt").write_text(text, encoding="utf-8")