"""3.3 - 3.5: train VisionGuard with MLflow tracking on DagsHub (manual logging or autolog)."""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import keras
import mlflow
import mlflow.keras
import mlflow.tensorflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature

from backbones import build_model, count_trainable, set_unfrozen
from eval_utils import choose_threshold, confusion_figure, metrics_at, roc_figure
from input_pipeline import SPLIT_DIR, build_dataset
from models import compile_model
from tracking import setup_tracking
from train_utils import class_weights, make_callbacks, set_seed


def retry(fn, *args, **kwargs):
    """Uploads to DagsHub can fail on a bad connection: try again before giving up."""
    for i in (1, 2):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"  upload failed ({type(e).__name__}), retry {i}/2 ...")
            time.sleep(5)
    return fn(*args, **kwargs)


def git_info():
    run = lambda *c: subprocess.check_output(["git", *c], text=True).strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))


def dvc_data_md5():
    return yaml.safe_load(open("data/raw.dvc"))["outs"][0]["md5"]


class LogEpochs(keras.callbacks.Callback):
    """Manual logging: every per-epoch number that Keras reports goes to MLflow."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", choices=["manual", "autolog"], default="manual")
    p.add_argument("--experiment", default="visionguard")
    p.add_argument("--target", choices=["dagshub", "local"], default="dagshub")
    p.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--unfrozen", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    set_seed(a.seed)
    setup_tracking(a.experiment, a.target)
    if a.mode == "autolog":
        try:
            mlflow.tensorflow.autolog(log_datasets=False, checkpoint=False)
        except TypeError:
            mlflow.tensorflow.autolog(log_datasets=False)

    train_ds = build_dataset("train", training=True, img_size=a.img_size, augment=not a.no_augment, seed=a.seed)
    val_ds = build_dataset("val", training=False, img_size=a.img_size)
    test_ds = build_dataset("test", training=False, img_size=a.img_size)
    cw = class_weights()

    model, base = build_model(a.backbone, a.img_size, a.dropout)
    set_unfrozen(base, a.unfrozen)
    compile_model(model, learning_rate=a.lr)

    commit, dirty = git_info()
    with mlflow.start_run(run_name=a.run_name) as run:
        mlflow.set_tags({"git_commit": commit, "git_dirty": str(dirty), "dvc_data_md5": dvc_data_md5(),
                         "seed": str(a.seed), "logging_mode": a.mode, "task": "ai_vs_real_classification"})
        if a.mode == "manual":
            mlflow.log_params({
                "backbone": a.backbone, "img_size": a.img_size, "batch_size": 32, "learning_rate": a.lr,
                "dropout": a.dropout, "unfrozen_layers": a.unfrozen, "augmentation": not a.no_augment,
                "max_epochs": a.epochs, "patience": a.patience, "optimizer": "adam",
                "loss": "binary_crossentropy", "seed": a.seed,
                "class_weight_real": round(cw[0], 3), "class_weight_ai": round(cw[1], 3),
                "trainable_params": count_trainable(model)})

        callbacks = make_callbacks(a.run_name, a.patience)
        if a.mode == "manual":
            callbacks.append(LogEpochs())
        hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=callbacks,
                         class_weight=cw, verbose=2)
        best = int(np.argmin(hist.history["val_loss"])) + 1   # weights of this epoch are restored

        # ---- final evaluation: threshold from validation, test metrics for the report ----
        val_df = pd.read_csv(SPLIT_DIR / "val.csv")
        test_df = pd.read_csv(SPLIT_DIR / "test.csv")
        vprob = model.predict(val_ds, verbose=0).ravel()
        tprob = model.predict(test_ds, verbose=0).ravel()
        yv, yt = val_df["label"].values, test_df["label"].values
        thr = choose_threshold(yv, vprob)
        vm, tm = metrics_at(yv, vprob, thr), metrics_at(yt, tprob, thr)
        final = {"best_epoch": best, "threshold": thr}
        final.update({f"val_final_{k}": v for k, v in vm.items()})
        final.update({f"test_{k}": v for k, v in tm.items()})
        mlflow.log_metrics({k: float(v) for k, v in final.items()})

        # ---- artifacts (3.4) ----
        retry(mlflow.log_figure, confusion_figure(yt, tprob, thr, a.run_name), "confusion_matrix.png")
        retry(mlflow.log_figure, roc_figure(yt, tprob, thr, a.run_name), "roc_curve.png")
        retry(mlflow.log_artifacts, str(SPLIT_DIR), "split_lists")
        with tempfile.TemporaryDirectory() as tmp:
            freeze = Path(tmp) / "pip_freeze.txt"
            freeze.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
            retry(mlflow.log_artifact, str(freeze))
        if a.mode == "manual":   # in autolog mode the model is logged by autolog itself
            xb, _ = next(iter(val_ds))
            sig = infer_signature(xb.numpy()[:2], model.predict(xb[:2], verbose=0))
            retry(mlflow.keras.log_model, model, name="model", signature=sig)

        print("\n" + "=" * 60)
        print(f"RUN {a.run_name} | mode {a.mode} | id {run.info.run_id}")
        print(f"best epoch {best} | threshold {thr:.2f} | val AUC {vm['roc_auc']:.3f} | "
              f"test accuracy {tm['accuracy']:.3f} | test AUC {tm['roc_auc']:.3f}")
        print(f"tags: git {commit[:8]} dirty={dirty} | data md5 {dvc_data_md5()[:8]}")


if __name__ == "__main__":
    main()