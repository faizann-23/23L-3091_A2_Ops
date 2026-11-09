"""2.5 Evaluation: choose thresholds on validation, evaluate on the test set ONCE."""
import sys
from pathlib import Path

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from input_pipeline import SPLIT_DIR, build_dataset
from train_utils import SEED, set_seed

REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
RESULT_FILE = REPORT_DIR / "test_metrics.csv"
MODELS = {"cnn": "models/cnn_best.keras", "mobilenet": "models/mobilenet_best.keras"}
NICE = {"cnn": "Baseline CNN", "mobilenet": "MobileNetV2 (transfer)"}

if RESULT_FILE.exists() and "--force" not in sys.argv:
    sys.exit("Test metrics already exist: the test set is meant to be used once. "
             "Use --force only if you really must repeat it.")

set_seed()
FIG_DIR.mkdir(parents=True, exist_ok=True)
lines = []


def say(t=""):
    print(t)
    lines.append(str(t))


def predict(model, split):
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    prob = model.predict(build_dataset(split, training=False), verbose=0).ravel()
    assert len(prob) == len(df), "prediction count does not match split list"
    return df, prob


def choose_threshold(y, prob):
    """Pick the threshold on VALIDATION data by maximising a smoothed Youden's J."""
    rows = []
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        recall, spec = tp / (tp + fn), tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        rows.append({"threshold": t, "recall": recall, "specificity": spec,
                     "precision": precision, "youden_j": recall + spec - 1})
    table = pd.DataFrame(rows)
    table["j_smooth"] = table["youden_j"].rolling(5, center=True, min_periods=1).mean()
    thr = float(table.loc[table["j_smooth"].idxmax(), "threshold"])
    return thr, table


def metrics_at(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(thr, 2),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "specificity": tn / (tn + fp),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, prob),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_ci(y, prob, thr, n=1000):
    rng = np.random.default_rng(SEED)
    accs, aucs = [], []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < 2:
            continue
        accs.append(accuracy_score(y[s], (prob[s] >= thr).astype(int)))
        aucs.append(roc_auc_score(y[s], prob[s]))
    return np.percentile(accs, [2.5, 97.5]), np.percentile(aucs, [2.5, 97.5])


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Real", "AI"])
    ax.set_yticks([0, 1], ["Real", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


rows, rocs, val_tables, thresholds = [], {}, {}, {}
for key, path in MODELS.items():
    model = keras.models.load_model(path)

    # 1) threshold from VALIDATION data only
    vdf, vprob = predict(model, "val")
    thr, vtable = choose_threshold(vdf["label"].values, vprob)
    vtable.to_csv(REPORT_DIR / f"threshold_validation_{key}.csv", index=False)
    val_tables[key], thresholds[key] = vtable, thr

    # 2) the single pass over the TEST set
    tdf, prob = predict(model, "test")
    y = tdf["label"].values
    acc_ci, auc_ci = bootstrap_ci(y, prob, thr)
    main = metrics_at(y, prob, thr)
    main.update(model=NICE[key], setting="chosen threshold",
                accuracy_95ci=f"{acc_ci[0]:.3f}-{acc_ci[1]:.3f}",
                auc_95ci=f"{auc_ci[0]:.3f}-{auc_ci[1]:.3f}")
    ref = metrics_at(y, prob, 0.5)
    ref.update(model=NICE[key], setting="default 0.5 (reference)", accuracy_95ci="", auc_95ci="")
    rows += [main, ref]

    tdf = tdf.assign(prob_ai=prob, pred=(prob >= thr).astype(int))
    tdf["correct"] = tdf["pred"] == tdf["label"]
    tdf.to_csv(REPORT_DIR / f"test_predictions_{key}.csv", index=False)

    cm = confusion_matrix(y, tdf["pred"], labels=[0, 1])
    plot_cm(cm, f"{NICE[key]}\nthreshold = {thr:.2f}", FIG_DIR / f"confusion_matrix_{key}.png")
    fpr, tpr, _ = roc_curve(y, prob)
    rocs[key] = (fpr, tpr, main["roc_auc"], main["recall"], 1 - main["specificity"])

# ROC curves, both models in one figure
fig, ax = plt.subplots(figsize=(5.6, 5.2))
for key, (fpr, tpr, auc, op_tpr, op_fpr) in rocs.items():
    line, = ax.plot(fpr, tpr, label=f"{NICE[key]} (AUC = {auc:.3f})")
    ax.scatter([op_fpr], [op_tpr], color=line.get_color(), zorder=5, s=45)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
ax.set_xlabel("False positive rate (real photo called AI)")
ax.set_ylabel("True positive rate (AI image caught)")
ax.set_title("ROC curves on the test set (dots = chosen thresholds)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "roc_curves.png", dpi=150)
plt.close(fig)

# threshold selection figure (validation data)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, key in zip(axes, MODELS):
    t = val_tables[key]
    ax.plot(t["threshold"], t["recall"], label="recall (AI caught)")
    ax.plot(t["threshold"], t["specificity"], label="specificity (real kept real)")
    ax.plot(t["threshold"], t["precision"], label="precision")
    ax.axvline(thresholds[key], color="k", linestyle="--", label=f"chosen = {thresholds[key]:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_title(f"{NICE[key]} (validation data)")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "threshold_selection.png", dpi=150)
plt.close(fig)

# summary tables
result = pd.DataFrame(rows)
cols = ["model", "setting", "threshold", "accuracy", "precision", "recall", "specificity",
        "f1", "roc_auc", "tn", "fp", "fn", "tp", "accuracy_95ci", "auc_95ci"]
result = result[cols]
result.to_csv(RESULT_FILE, index=False)

n_test = int(rows[0]["tn"] + rows[0]["fp"] + rows[0]["fn"] + rows[0]["tp"])
share_ai = (rows[0]["fn"] + rows[0]["tp"]) / n_test
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
say("=" * 70)
say(f"TEST SET EVALUATION  (n = {n_test}, share of AI images = {share_ai:.3f})")
say(f"Majority-class baseline accuracy: {max(share_ai, 1 - share_ai):.3f}")
say("=" * 70)
say(result.round(3).to_string(index=False))
say()
for key in MODELS:
    say(f"Threshold chosen on validation for {NICE[key]}: {thresholds[key]:.2f}")
(REPORT_DIR / "evaluation_summary.txt").write_text("\n".join(lines), encoding="utf-8")