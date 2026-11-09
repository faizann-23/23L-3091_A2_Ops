"""3.5: collect the manual VisionGuard runs from DagsHub into one table and one chart."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from tracking import setup_tracking

CHANGE = {
    "A_mnv2_base_manual": "baseline (MobileNetV2, lr 1e-3, frozen, augmentation on, 224 px)",
    "B_mnv2_lr1e-4": "learning rate 1e-3 -> 1e-4 (vs A)",
    "C_mnv2_no_aug": "augmentation switched off (vs A)",
    "D_mnv2_img160": "image size 224 -> 160 px (vs A)",
    "E_mnv2_unfrozen40": "last 40 backbone layers unfrozen (vs B)",
    "F_effb0_lr1e-4": "backbone MobileNetV2 -> EfficientNetB0 (vs B)",
}

setup_tracking("visionguard", "dagshub")
exp = mlflow.get_experiment_by_name("visionguard")
runs = mlflow.search_runs([exp.experiment_id], filter_string="attributes.status = 'FINISHED'",
                          order_by=["attributes.start_time ASC"])
client = MlflowClient()

rows = []
for _, r in runs.iterrows():
    name = r.get("tags.mlflow.runName")
    hist = client.get_metric_history(r["run_id"], "val_loss")
    rows.append({
        "run": name,
        "mode": r.get("tags.logging_mode"),
        "run_id": r["run_id"],
        "change": CHANGE.get(name, ""),
        "best_epoch": r.get("metrics.best_epoch"),
        "best_val_loss": min((m.value for m in hist), default=np.nan),
        "val_auc": r.get("metrics.val_final_roc_auc"),
        "val_accuracy": r.get("metrics.val_final_accuracy"),
        "threshold": r.get("metrics.threshold"),
        "test_auc": r.get("metrics.test_roc_auc"),
        "test_accuracy": r.get("metrics.test_accuracy"),
        "test_f1": r.get("metrics.test_f1"),
    })
df = pd.DataFrame(rows).drop_duplicates("run", keep="last")   # a re-run replaces the older run of the same name
manual = df[df["mode"] == "manual"].copy()
manual = manual.sort_values(["val_auc", "best_val_loss"], ascending=[False, True]).reset_index(drop=True)

Path("reports/figures").mkdir(parents=True, exist_ok=True)
manual.drop(columns="run_id").to_csv("reports/run_comparison.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
print("=" * 70)
print("RUN COMPARISON (manual runs, sorted by validation AUC; test columns are for reporting only)")
print("=" * 70)
print(manual.drop(columns=["run_id", "mode"]).round(4).to_string(index=False))
auto = df[df["mode"] == "autolog"]
if len(auto):
    print("\nAutolog run (same config as A):")
    print(auto[["run", "best_epoch", "val_auc", "test_auc", "test_accuracy"]].round(4).to_string(index=False))

best = manual.iloc[0]
print(f"\nBEST by validation AUC: {best['run']}  (val AUC {best['val_auc']:.4f}, "
      f"best val loss {best['best_val_loss']:.4f})")

base = manual[manual["run"] == "A_mnv2_base_manual"]
cand = {"best": {"run": best["run"], "run_id": best["run_id"]}}
if len(base):
    cand["baseline"] = {"run": base.iloc[0]["run"], "run_id": base.iloc[0]["run_id"]}
Path("reports/registry_candidates.json").write_text(json.dumps(cand, indent=2))

# chart
labels = [str(n).split("_")[0] for n in manual["run"]]
x = np.arange(len(manual))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - 0.2, manual["val_auc"], 0.4, label="validation AUC (used to choose)")
ax.bar(x + 0.2, manual["test_auc"], 0.4, label="test AUC (report only)")
ax.set_xticks(x, labels)
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0.5, 1.0)
ax.set_title("VisionGuard runs (A = baseline), sorted by validation AUC")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig("reports/figures/run_comparison.png", dpi=150)
print("Saved reports/run_comparison.csv, reports/registry_candidates.json, reports/figures/run_comparison.png")