"""2.7 Apply part: Keras Tuner random search for VisionGuard (MobileNetV2)."""
import json
from pathlib import Path

import keras
import keras_tuner as kt
import numpy as np
import pandas as pd
from keras import layers

from input_pipeline import build_dataset
from models import compile_model
from train_utils import SEED, class_weights, set_seed
from transfer import build_mobilenet

set_seed()
REPORT_DIR = Path("reports")
N_TRIALS = 10

train_ds = build_dataset("train", training=True)   # decoded images are cached after the first epoch
val_ds = build_dataset("val", training=False)
cw = class_weights()


def build(hp):
    lr = hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
    dropout = hp.Float("dropout", 0.1, 0.6, step=0.1)
    n_unfrozen = hp.Choice("unfrozen_layers", [0, 20, 40, 80])

    model, base = build_mobilenet(dropout=dropout)
    base.trainable = True
    top = base.layers[-n_unfrozen:] if n_unfrozen > 0 else []
    top_ids = {id(l) for l in top}
    for layer in base.layers:  # BatchNorm layers always stay frozen
        layer.trainable = (id(layer) in top_ids) and not isinstance(layer, layers.BatchNormalization)
    return compile_model(model, learning_rate=lr)


tuner = kt.RandomSearch(build, objective="val_loss", max_trials=N_TRIALS, seed=SEED,
                        directory="tuner_logs", project_name="visionguard", overwrite=True)
stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
tuner.search(train_ds, validation_data=val_ds, epochs=12, class_weight=cw, callbacks=[stop])

rows = []
for t in tuner.oracle.get_best_trials(num_trials=N_TRIALS):
    try:
        auc = float(t.metrics.get_best_value("val_auc"))
    except Exception:
        auc = float("nan")
    rows.append({"trial": t.trial_id,
                 "learning_rate": round(float(t.hyperparameters.get("learning_rate")), 6),
                 "dropout": round(float(t.hyperparameters.get("dropout")), 2),
                 "unfrozen_layers": int(t.hyperparameters.get("unfrozen_layers")),
                 "best_val_loss": round(float(t.score), 4),
                 "best_val_auc": round(auc, 4)})
df = pd.DataFrame(rows)
df.to_csv(REPORT_DIR / "tuner_visionguard_results.csv", index=False)

best = tuner.get_best_hyperparameters(1)[0]
(REPORT_DIR / "tuner_best.json").write_text(json.dumps(best.values, indent=2))

df["log10_lr"] = np.log10(df["learning_rate"])
corr = (df[["log10_lr", "dropout", "unfrozen_layers", "best_val_loss"]]
        .corr(method="spearman")["best_val_loss"].drop("best_val_loss").round(2))
by_unf = df.groupby("unfrozen_layers")["best_val_loss"].agg(["count", "mean", "min"]).round(4)

pd.set_option("display.width", 160)
text = ("VISIONGUARD TUNING RESULTS (random search, best validation loss first)\n\n"
        + df.drop(columns="log10_lr").to_string(index=False)
        + f"\n\nBest settings: {best.values}"
        + "\n\nSpearman correlation with best_val_loss (negative = larger value helps):\n"
        + corr.to_string()
        + "\n\nMean best_val_loss by number of unfrozen layers:\n"
        + by_unf.to_string())
print("\n" + text)
(REPORT_DIR / "tuner_visionguard_summary.txt").write_text(text, encoding="utf-8")