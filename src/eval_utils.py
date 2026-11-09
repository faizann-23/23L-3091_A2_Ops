"""Threshold selection, metrics and figures used by the MLflow training script."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def choose_threshold(y, prob):
    """Threshold chosen on VALIDATION data: maximise a smoothed Youden's J (recall + specificity - 1)."""
    ts = np.round(np.arange(0.05, 0.951, 0.01), 2)
    j = []
    for t in ts:
        tn, fp, fn, tp = confusion_matrix(y, (prob >= t).astype(int), labels=[0, 1]).ravel()
        j.append(tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1)
    smooth = pd.Series(j).rolling(5, center=True, min_periods=1).mean().values
    return float(ts[int(np.argmax(smooth))])


def metrics_at(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "specificity": tn / max(tn + fp, 1),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, prob)}


def confusion_figure(y, prob, thr, title):
    cm = confusion_matrix(y, (prob >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Real", "AI"])
    ax.set_yticks([0, 1], ["Real", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for k in range(2):
            ax.text(k, i, int(cm[i, k]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, k] > cm.max() / 2 else "black")
    ax.set_title(f"{title}\nthreshold = {thr:.2f}", fontsize=10)
    fig.tight_layout()
    return fig


def roc_figure(y, prob, thr, title):
    fpr, tpr, _ = roc_curve(y, prob)
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, prob):.3f}")
    ax.scatter([fp / max(fp + tn, 1)], [tp / max(tp + fn, 1)], color="red", zorder=5, label="chosen threshold")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve (test set): {title}", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig