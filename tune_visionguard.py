import os
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# Set seeds
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

def run_toy_fashion_mnist_tuner():
    """
    Sub-task 2.7 Toy Part:
    Tune a tiny network on Fashion-MNIST with Hyperband / RandomSearch.
    Searching: learning rate and units.
    """
    print("\n" + "="*80)
    print("SUB-TASK 2.7 TOY PART: KERAS TUNER ON FASHION-MNIST")
    print("="*80)

    # Load Fashion-MNIST subset for fast demonstration
    (x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:2000].astype("float32") / 255.0
    y_train = y_train[:2000]
    x_val = x_val[:500].astype("float32") / 255.0
    y_val = y_val[:500]

    def build_toy_model(hp):
        model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(
                units=hp.Int("units", min_value=32, max_value=128, step=32),
                activation="relu"
            ),
            layers.Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)),
            layers.Dense(10, activation="softmax")
        ])
        lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    tuner = kt.Hyperband(
        build_toy_model,
        objective="val_accuracy",
        max_epochs=4,
        factor=3,
        directory="tuner_toy_dir",
        project_name="fashion_mnist_tuning",
        overwrite=True
    )

    tuner.search(x_train, y_train, epochs=4, validation_data=(x_val, y_val), verbose=0)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n[+] Fashion-MNIST Toy Tuner Best Hyperparameters:")
    print(f"  - Units        : {best_hps.get('units')}")
    print(f"  - Dropout      : {best_hps.get('dropout')}")
    print(f"  - Learning Rate: {best_hps.get('learning_rate')}")

    # Results summary table
    trials = tuner.oracle.get_best_trials(num_trials=5)
    records = []
    for t in trials:
        records.append({
            "Trial ID": t.trial_id,
            "Units": t.hyperparameters.get("units"),
            "Dropout": t.hyperparameters.get("dropout"),
            "Learning Rate": t.hyperparameters.get("learning_rate"),
            "Val Accuracy": round(t.score, 4)
        })
    toy_summary = pd.DataFrame(records)
    print("\nTOY TUNER TOP TRIALS:")
    print(toy_summary.to_string(index=False))
    toy_summary.to_csv("tuner_toy_results.csv", index=False)
    return toy_summary

def run_visionguard_tuner(audit_csv="dataset_audit.csv"):
    """
    Sub-task 2.7 Apply Part:
    Tune VisionGuard model with at least 8 trials.
    Searching: learning_rate, dropout, and units.
    """
    print("\n" + "="*80)
    print("SUB-TASK 2.7 APPLY PART: KERAS TUNER ON VISIONGUARD (8 TRIALS)")
    print("="*80)

    # Load splits
    train_df = pd.read_csv("train_split.csv")
    val_df = pd.read_csv("val_split.csv")

    from train_visionguard import build_dataset, IMG_SIZE
    train_ds = build_dataset(train_df, is_training=True)
    val_ds = build_dataset(val_df, is_training=False)

    def build_tune_model(hp):
        inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = layers.Rescaling(1./255)(inputs)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.GlobalAveragePooling2D()(x)
        
        dense_units = hp.Choice("units", values=[64, 128])
        x = layers.Dense(dense_units, activation="relu")(x)
        
        drop_rate = hp.Choice("dropout", values=[0.2, 0.3, 0.5])
        x = layers.Dropout(drop_rate)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        lr = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4])
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
        )
        return model

    tuner = kt.RandomSearch(
        build_tune_model,
        objective="val_accuracy",
        max_trials=8,
        seed=SEED,
        directory="tuner_visionguard_dir",
        project_name="visionguard_tuning",
        overwrite=True
    )

    tuner.search(train_ds, validation_data=val_ds, epochs=1, verbose=1)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n[+] VisionGuard Tuner Best Hyperparameters (out of 8 trials):")
    print(f"  - Units        : {best_hps.get('units')}")
    print(f"  - Dropout      : {best_hps.get('dropout')}")
    print(f"  - Learning Rate: {best_hps.get('learning_rate')}")

    # Results summary table for report
    trials = tuner.oracle.get_best_trials(num_trials=8)
    records = []
    for rank, t in enumerate(trials, 1):
        records.append({
            "Rank": rank,
            "Trial ID": t.trial_id,
            "Units": t.hyperparameters.get("units"),
            "Dropout": t.hyperparameters.get("dropout"),
            "Learning Rate": t.hyperparameters.get("learning_rate"),
            "Val Accuracy": round(t.score, 4)
        })
    vg_summary = pd.DataFrame(records)
    print("\nVISIONGUARD 8-TRIAL TUNING RESULTS TABLE:")
    print(vg_summary.to_string(index=False))
    vg_summary.to_csv("tuner_visionguard_results.csv", index=False)
    return vg_summary

if __name__ == "__main__":
    if not os.path.exists("tuner_toy_results.csv"):
        run_toy_fashion_mnist_tuner()
    run_visionguard_tuner()

