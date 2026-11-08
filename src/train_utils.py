"""Shared training helpers: seeds, class weights, callbacks, curves, summaries."""
import json
from pathlib import Path

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"


def set_seed(seed=SEED):
    keras.utils.set_random_seed(seed)  # seeds python, numpy and TensorFlow


def class_weights(split_csv="data/splits/train.csv"):
    y = pd.read_csv(split_csv)["label"].values
    n, n_ai = len(y), int(y.sum())
    n_real = n - n_ai
    return {0: n / (2 * n_real), 1: n / (2 * n_ai)}


def make_callbacks(name, patience=8, lr_patience=3):
    MODEL_DIR.mkdir(exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / f"{name}_best.keras"), monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=lr_patience, min_lr=1e-7, verbose=1),
    ]


def plot_history(h, best, title, out_path):
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(epochs, h[key], label="train")
        ax.plot(epochs, h["val_" + key], label="validation")
        ax.axvline(best, color="green", linestyle="--", label=f"best epoch = {best}")
        ax.scatter([best], [h["val_" + key][best - 1]], color="green", zorder=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{title}: {label}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_and_report(model, name, train_ds, val_ds, epochs, patience=8,
                   class_weight=None, title=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=make_callbacks(name, patience),
                     class_weight=class_weight, verbose=2)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    pd.DataFrame(h).assign(epoch=lambda d: d.index + 1).to_csv(
        REPORT_DIR / f"{name}_history.csv", index=False)

    best = int(np.argmin(h["val_loss"])) + 1
    i = best - 1
    plot_history(h, best, title or name, FIG_DIR / f"{name}_curves.png")
    summary = {
        "name": name,
        "epochs_run": len(h["loss"]),
        "best_epoch": best,
        "train_loss": h["loss"][i],
        "train_accuracy": h["accuracy"][i],
        "val_loss": h["val_loss"][i],
        "val_accuracy": h["val_accuracy"][i],
        "val_auc": h["val_auc"][i],
        "learning_rate_at_best": h.get("learning_rate", [None] * len(h["loss"]))[i],
    }
    (REPORT_DIR / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[{name}] epochs run: {summary['epochs_run']} | best epoch: {best} | "
          f"val_loss {summary['val_loss']:.4f} | val_acc {summary['val_accuracy']:.4f} | "
          f"val_auc {summary['val_auc']:.4f}")
    return summary