"""Tasks 3.3 - 3.6: train every configuration for real, track it in MLflow (DagsHub), compare, register.

What this script does
  1. Trains 7 runs. Every run builds its own model from its own configuration, so the
     parameters stored in MLflow are exactly what was used (nothing is only a label).
  2. Run 1 logs by hand (a Keras callback), run 2 repeats run 1 with MLflow autologging (3.3).
  3. Per run: parameters, per-epoch metrics, final metrics, tags (git commit, DVC data hash,
     seed) and artifacts (confusion matrix, ROC curve, split lists, package list, model) (3.4).
  4. The decision threshold is chosen on the VALIDATION set only. The test set is used once
     per run, for the final numbers. The champion is picked by VALIDATION ROC-AUC, never by
     test score, so the test set stays an honest estimate.
  5. Writes the comparison table / chart (3.5) and registers 2 versions, alias "champion" (3.6).

Usage
    python log_visionguard_experiments.py              # full run
    python log_visionguard_experiments.py --only run1_baseline_manual run3_baseline_lr_3e4
    python log_visionguard_experiments.py --no-register
Secrets come from environment variables / .env (DAGSHUB_USER_TOKEN), never from this file.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.tensorflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import tensorflow as tf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

load_dotenv()

SEED = 42
BATCH_SIZE = 32
STUDENT_ID = "23L-3091"
EXPERIMENT_NAME = f"VisionGuard_A2_Ops_{STUDENT_ID.replace('-', '_')}"
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "visionguard-detector")
ARTIFACT_DIR = "artifacts"          # git-ignored: run checkpoints live here
REPORT_DIR = "reports"

# --------------------------------------------------------------------------------------
# The 7 runs. "differs from" tells you the ONE change versus the reference (run 1).
# --------------------------------------------------------------------------------------
RUNS = [
    dict(name="run1_baseline_manual", backbone="baseline_cnn", lr=1e-3, img=128, aug="light",
         dropout=0.4, logging="manual", finetune=False,
         differs_from="reference (baseline CNN, lr 1e-3, light augmentation, 128px)"),
    dict(name="run2_baseline_autolog", backbone="baseline_cnn", lr=1e-3, img=128, aug="light",
         dropout=0.4, logging="autolog", finetune=False,
         differs_from="run1: only the logging method (autologging instead of a callback)"),
    dict(name="run3_baseline_lr_3e4", backbone="baseline_cnn", lr=3e-4, img=128, aug="light",
         dropout=0.4, logging="manual", finetune=False,
         differs_from="run1: learning rate 3e-4"),
    dict(name="run4_baseline_heavy_aug", backbone="baseline_cnn", lr=1e-3, img=128, aug="heavy",
         dropout=0.4, logging="manual", finetune=False,
         differs_from="run1: heavy augmentation"),
    dict(name="run5_baseline_img96", backbone="baseline_cnn", lr=1e-3, img=96, aug="light",
         dropout=0.4, logging="manual", finetune=False,
         differs_from="run1: image size 96x96"),
    dict(name="run6_mobilenetv2_frozen", backbone="mobilenetv2", lr=1e-3, img=128, aug="light",
         dropout=0.3, logging="manual", finetune=False,
         differs_from="run1: backbone (MobileNetV2, whole base frozen)"),
    dict(name="run7_mobilenetv2_finetune", backbone="mobilenetv2", lr=1e-3, img=128, aug="light",
         dropout=0.3, logging="manual", finetune=True, finetune_lr=1e-5,
         differs_from="run6: last block unfrozen and trained with lr 1e-5"),
]


# --------------------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------------------
def configure_tracking():
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "faizann-23")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "23L-3091_A2_Ops")
    uri = os.getenv("MLFLOW_TRACKING_URI", f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    if "dagshub.com" in uri:
        token = os.getenv("DAGSHUB_USER_TOKEN")
        if not token:
            sys.exit("DAGSHUB_USER_TOKEN is not set. Put it in .env (see .env.example).")
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", repo_owner)
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri, repo_owner, repo_name


def get_git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        return sha, bool(dirty)
    except Exception:
        return "unknown", False


def get_dvc_hash():
    """Dataset version = md5 stored in whichever .dvc pointer file the repo has (outside .dvc/)."""
    candidates = ["dataset.dvc", "data.dvc"] + sorted(glob.glob("data/*.dvc")) + \
        sorted(glob.glob("**/*.dvc", recursive=True))
    for path in candidates:
        if not os.path.isfile(path) or path.replace("\\", "/").startswith(".dvc"):
            continue
        with open(path, "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split("md5:")[1].strip()
    return "unknown"


def safe_log_params(params):
    """Log parameters one by one. If autologging already stored a key, keep going."""
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except Exception as e:  # noqa: BLE001
            print(f"  [!] param '{k}' not logged ({str(e)[:70]})")


def set_seeds():
    keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)


# --------------------------------------------------------------------------------------
# Data (same pipeline idea as Task 2.1: read paths -> decode -> resize -> batch -> prefetch)
# --------------------------------------------------------------------------------------
def load_splits():
    frames = {}
    for name in ("train", "val", "test"):
        path = f"{name}_split.csv"
        if not os.path.exists(path):
            sys.exit(f"{path} not found. Run audit_and_eda.py and train_visionguard.py first.")
        frames[name] = pd.read_csv(path)
    missing = sum(not os.path.exists(p) for df in frames.values() for p in df["filepath"])
    if missing:
        sys.exit(f"{missing} image paths in the split CSVs do not exist. "
                 f"Check the dataset/ junction and regenerate the splits.")
    return frames["train"], frames["val"], frames["test"]


def augmentation(kind):
    if kind == "heavy":
        return keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.25),
                                 layers.RandomZoom(0.2), layers.RandomContrast(0.2)])
    return keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),
                             layers.RandomZoom(0.1)])


def make_dataset(df, img, train=False, aug="light"):
    size = (img, img)
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label"].values.astype("float32")))

    def load(path, label):
        data = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(data, size), label    # float32 in [0, 255], aspect ratio not kept (as in serving)

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if train:                                         # augmentation on training data only
        aug_layer = augmentation(aug)
        ds = ds.shuffle(len(df), seed=SEED).map(lambda x, y: (aug_layer(x, training=True), y),
                                                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])


def build_baseline(img, dropout, lr):
    inputs = keras.Input(shape=(img, img, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)          # scaling lives INSIDE the model -> serving matches
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="baseline_cnn")
    compile_model(model, lr)
    return model, None


def build_mobilenet(img, dropout, lr, pretrained=True):
    base = keras.applications.MobileNetV2(input_shape=(img, img, 3), include_top=False,
                                          weights="imagenet" if pretrained else None)
    base.trainable = False
    inputs = keras.Input(shape=(img, img, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)   # scales [0,255] -> [-1,1] inside the model
    x = base(x, training=False)                                   # BatchNorm stays in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    compile_model(model, lr)
    return model, base


def unfreeze_last_block(base):
    """Unfreeze only the last inverted-residual block (block_16_*) plus the final 1x1 conv.
    BatchNorm layers stay frozen so their statistics are not disturbed by the small dataset."""
    names = [l.name for l in base.layers]
    start = names.index("block_16_expand") if "block_16_expand" in names else len(names) - 25
    base.trainable = True
    for l in base.layers[:start]:
        l.trainable = False
    for l in base.layers[start:]:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    return [l.name for l in base.layers if l.trainable]


# --------------------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------------------
class MLflowEpochCallback(keras.callbacks.Callback):
    """Manual logging: every Keras metric of every epoch, plus the learning rate."""

    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        step = self.offset + epoch
        for k, v in (logs or {}).items():
            mlflow.log_metric(k, float(v), step=step)
        try:
            mlflow.log_metric("learning_rate_epoch",
                              float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)), step=step)
        except Exception:  # noqa: BLE001
            pass


def callbacks_for(ckpt_path, manual, offset=0):
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
           keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)]
    if manual:
        cbs.append(MLflowEpochCallback(offset))
    return cbs


def select_threshold(y_true, prob):
    """Threshold with the best F1 on the VALIDATION set (ties -> closest to 0.5)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in grid]
    best = max(scores)
    candidates = [t for t, s in zip(grid, scores) if abs(s - best) < 1e-9]
    return float(min(candidates, key=lambda t: abs(t - 0.5))), grid, scores


def metrics_at(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    return dict(accuracy=accuracy_score(y_true, pred),
                precision=precision_score(y_true, pred, zero_division=0),
                recall=recall_score(y_true, pred, zero_division=0),
                f1=f1_score(y_true, pred, zero_division=0),
                roc_auc=roc_auc_score(y_true, prob))


def save_eval_figures(run_name, y_true, prob, thr):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    cm_path = os.path.join(ARTIFACT_DIR, f"{run_name}_confusion_matrix.png")
    roc_path = os.path.join(ARTIFACT_DIR, f"{run_name}_roc_curve.png")
    cm = confusion_matrix(y_true, (prob >= thr).astype(int), labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "AI"], yticklabels=["Real", "AI"])
    plt.title(f"{run_name}: test confusion matrix (threshold {thr:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    fpr, tpr, _ = roc_curve(y_true, prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_true, prob):.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{run_name}: test ROC curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    return cm_path, roc_path


def package_list():
    path = os.path.join(ARTIFACT_DIR, "pip_freeze.txt")
    with open(path, "w") as f:
        f.write(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode())
    return path


def run_one(cfg, data, tags, args):
    train_df, val_df, test_df = data
    img = cfg["img"]
    manual = cfg["logging"] == "manual"
    os.makedirs(os.path.join(ARTIFACT_DIR, "models"), exist_ok=True)
    ckpt = os.path.join(ARTIFACT_DIR, "models", f"{cfg['name']}.keras")

    train_ds = make_dataset(train_df, img, train=True, aug=cfg["aug"])
    val_ds = make_dataset(val_df, img)
    test_ds = make_dataset(test_df, img)
    if args.quick:                                    # smoke test only (never for real results)
        train_ds, val_ds = train_ds.take(2), val_ds.take(1)

    set_seeds()
    if cfg["backbone"] == "baseline_cnn":
        model, base = build_baseline(img, cfg["dropout"], cfg["lr"])
        stage_epochs = [args.max_epochs]
    else:
        model, base = build_mobilenet(img, cfg["dropout"], cfg["lr"], pretrained=not args.no_pretrained)
        stage_epochs = [min(args.max_epochs, 8)] + ([min(args.max_epochs, 8)] if cfg["finetune"] else [])
    if args.quick:
        stage_epochs = [1 for _ in stage_epochs]

    frozen = "none" if cfg["backbone"] == "baseline_cnn" else "all backbone layers"
    params = {"backbone": cfg["backbone"], "learning_rate": cfg["lr"], "image_size": img,
              "augmentation": cfg["aug"], "dropout": cfg["dropout"], "train_batch_size": BATCH_SIZE,
              "optimizer": "Adam", "max_epochs_per_stage": stage_epochs[0], "seed": SEED,
              "frozen_layers": frozen, "finetune": cfg["finetune"], "logging_method": cfg["logging"],
              "differs_from": cfg["differs_from"][:250]}
    if cfg["finetune"]:
        params["finetune_lr"] = cfg["finetune_lr"]

    with mlflow.start_run(run_name=cfg["name"]) as run:
        if manual:
            safe_log_params(params)
        else:
            mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

        for k, v in tags.items():
            mlflow.set_tag(k, v)
        mlflow.set_tag("student_id", STUDENT_ID)
        mlflow.set_tag("logging_method", cfg["logging"])
        mlflow.set_tag("differs_from", cfg["differs_from"])

        print(f"  training stage 1 (max {stage_epochs[0]} epochs)...")
        hist = model.fit(train_ds, validation_data=val_ds, epochs=stage_epochs[0], verbose=2,
                         callbacks=callbacks_for(ckpt, manual))
        val_loss_hist = list(hist.history["val_loss"])
        offset = len(hist.history["loss"])

        if cfg["finetune"]:
            trainable = unfreeze_last_block(base)
            compile_model(model, cfg["finetune_lr"])      # re-compile so the new learning rate/trainable set apply
            params["frozen_layers"] = "all except last block (block_16_* + Conv_1); BatchNorm kept frozen"
            if manual:
                mlflow.set_tag("trainable_backbone_layers", ", ".join(trainable)[:480])
            with open(os.path.join(ARTIFACT_DIR, "trainable_layers.txt"), "w") as f:
                f.write("\n".join(trainable))
            print(f"  training stage 2: {len(trainable)} backbone layers unfrozen (lr {cfg['finetune_lr']})...")
            hist2 = model.fit(train_ds, validation_data=val_ds, epochs=stage_epochs[1], verbose=2,
                              callbacks=callbacks_for(ckpt, manual, offset=offset))
            val_loss_hist += list(hist2.history["val_loss"])

        if not manual:
            safe_log_params(params)                         # autolog filled its own params first; add ours
            mlflow.autolog(disable=True)

        best_epoch = int(np.argmin(val_loss_hist)) + 1
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_loss", float(np.min(val_loss_hist)))

        # Evaluation: model now holds the best weights (restore_best_weights)
        full_val = make_dataset(val_df, img)
        val_prob = model.predict(full_val, verbose=0).ravel()
        thr, grid, f1_curve = select_threshold(val_df["label"].values, val_prob)   # VALIDATION only
        val_m = metrics_at(val_df["label"].values, val_prob, thr)
        test_prob = model.predict(test_ds, verbose=0).ravel()                      # test used once
        test_m = metrics_at(test_df["label"].values, test_prob, thr)

        mlflow.log_param("decision_threshold", thr)
        for k, v in val_m.items():
            mlflow.log_metric(f"val_final_{k}", float(v))
        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", float(v))

        # Artifacts (3.4)
        cm_path, roc_path = save_eval_figures(cfg["name"], test_df["label"].values, test_prob, thr)
        for split in ("train", "val", "test"):
            mlflow.log_artifact(f"{split}_split.csv", artifact_path="splits")
        mlflow.log_artifact(package_list(), artifact_path="environment")
        mlflow.log_artifact(cm_path, artifact_path="evaluation")
        mlflow.log_artifact(roc_path, artifact_path="evaluation")
        if cfg["finetune"]:
            mlflow.log_artifact(os.path.join(ARTIFACT_DIR, "trainable_layers.txt"), artifact_path="evaluation")
        model.save(ckpt)                                    # best weights, final model file
        mlflow.log_artifact(ckpt, artifact_path="model_keras")

        print(f"  [+] run {run.info.run_id[:8]} | best epoch {best_epoch} | thr {thr:.2f} | "
              f"val AUC {val_m['roc_auc']:.4f} | test acc {test_m['accuracy']:.4f} | test AUC {test_m['roc_auc']:.4f}")
        return dict(cfg=cfg, run_id=run.info.run_id, ckpt=ckpt, thr=thr, best_epoch=best_epoch,
                    val=val_m, test=test_m, val_prob=val_prob, grid=grid, f1_curve=f1_curve,
                    val_loss=float(np.min(val_loss_hist)))


# --------------------------------------------------------------------------------------
# 3.5 comparison and 3.6 registry
# --------------------------------------------------------------------------------------
def comparison(results):
    rows = []
    for r in results:
        rows.append({"run": r["cfg"]["name"], "differs_from": r["cfg"]["differs_from"],
                     "best_epoch": r["best_epoch"], "threshold": r["thr"],
                     "val_roc_auc": round(r["val"]["roc_auc"], 4), "val_f1": round(r["val"]["f1"], 4),
                     "test_accuracy": round(r["test"]["accuracy"], 4), "test_precision": round(r["test"]["precision"], 4),
                     "test_recall": round(r["test"]["recall"], 4), "test_f1": round(r["test"]["f1"], 4),
                     "test_roc_auc": round(r["test"]["roc_auc"], 4), "run_id": r["run_id"]})
    df = pd.DataFrame(rows)
    os.makedirs(REPORT_DIR, exist_ok=True)
    csv_path = os.path.join(REPORT_DIR, "run_comparison.csv")
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["val_roc_auc"], 0.4, label="validation ROC-AUC (used to choose)")
    ax.bar(x + 0.2, df["test_roc_auc"], 0.4, label="test ROC-AUC (reported)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n", 1) for n in df["run"]], fontsize=8)
    ax.set_ylim(max(0.0, min(df["val_roc_auc"].min(), df["test_roc_auc"].min()) - 0.05), 1.0)
    ax.set_title("VisionGuard: run comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    chart_path = os.path.join(REPORT_DIR, "run_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return df, csv_path, chart_path


def threshold_figure(champ):
    path = os.path.join(REPORT_DIR, "champion_threshold_selection.png")
    y = champ["val_prob"]
    plt.figure(figsize=(6, 4))
    plt.plot(champ["grid"], champ["f1_curve"], lw=2, label="validation F1")
    plt.axvline(champ["thr"], color="red", ls="--", label=f"chosen threshold {champ['thr']:.2f}")
    plt.xlabel("decision threshold")
    plt.ylabel("F1 on validation set")
    plt.title(f"{champ['cfg']['name']}: threshold chosen on validation data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    del y
    return path


def register(results, df, csv_path, chart_path, thr_path):
    ranked = sorted(results, key=lambda r: (r["val"]["roc_auc"], r["val"]["f1"], -r["val_loss"]), reverse=True)
    champ = ranked[0]
    reference = next((r for r in results if r["cfg"]["name"] == RUNS[0]["name"]), None)
    first = reference if reference is not None and reference is not champ else ranked[1]
    client = MlflowClient()
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME,
                                       description="VisionGuard: AI-generated vs real image classifier")
    except Exception as e:  # noqa: BLE001
        print(f"[*] Registered model '{REGISTERED_MODEL_NAME}' already exists ({str(e)[:60]})")

    versions = {}
    for label, r in (("Version A: reference model", first), ("Version B: best validation ROC-AUC", champ)):
        mv = client.create_model_version(
            name=REGISTERED_MODEL_NAME, source=f"runs:/{r['run_id']}/model_keras", run_id=r["run_id"],
            description=f"{label} ({r['cfg']['name']}, val ROC-AUC {r['val']['roc_auc']:.4f})")
        versions[r["run_id"]] = mv.version
        print(f"[+] {label}: {REGISTERED_MODEL_NAME} v{mv.version} <- {r['cfg']['name']}")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", versions[champ["run_id"]])
    print(f"[+] Alias @champion -> version {versions[champ['run_id']]}")

    for p in (csv_path, chart_path, thr_path):
        client.log_artifact(champ["run_id"], p, artifact_path="comparison")

    shutil.copy(champ["ckpt"], "champion_model.keras")
    info = {"registered_model_name": REGISTERED_MODEL_NAME, "alias": "champion",
            "version": versions[champ["run_id"]], "run_id": champ["run_id"], "run_name": champ["cfg"]["name"],
            "image_size": champ["cfg"]["img"], "threshold": champ["thr"],
            "val_roc_auc": champ["val"]["roc_auc"], "test": {k: float(v) for k, v in champ["test"].items()}}
    with open("champion_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return champ, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="run names to execute (default: all 7)")
    parser.add_argument("--no-register", action="store_true", help="skip Task 3.6")
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--quick", action="store_true", help="smoke test: 1 epoch, 2 batches (never for real results)")
    parser.add_argument("--no-pretrained", action="store_true", help="smoke test only: skip ImageNet weights")
    args = parser.parse_args()

    uri, owner, repo = configure_tracking()
    if any(c["logging"] == "autolog" for c in RUNS if not args.only or c["name"] in args.only):
        try:
            import tensorboard  # noqa: F401  (MLflow's TensorFlow autologging writes TensorBoard summaries)
        except ImportError:
            sys.exit("MLflow autologging needs TensorBoard. Install it:  pip install tensorboard")
    sha, dirty = get_git_commit()
    dvc_hash = get_dvc_hash()
    tags = {"git_commit": sha, "git_dirty": str(dirty), "dvc_data_version": dvc_hash, "random_seed": str(SEED)}
    print("=" * 80)
    print(f"EXPERIMENT : {EXPERIMENT_NAME}\nTRACKING   : {uri}\nGIT COMMIT : {sha} (uncommitted changes: {dirty})"
          f"\nDVC DATA   : {dvc_hash}\nTF {tf.__version__} | MLflow {mlflow.__version__}")
    print("=" * 80)

    data = load_splits()
    print(f"Splits: train={len(data[0])} val={len(data[1])} test={len(data[2])}")
    selected = [c for c in RUNS if not args.only or c["name"] in args.only]
    results = []
    for i, cfg in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {cfg['name']}  ({cfg['differs_from']})")
        results.append(run_one(cfg, data, tags, args))

    df, csv_path, chart_path = comparison(results)
    print("\n" + "=" * 80 + "\nRUN COMPARISON (Sub-task 3.5)\n" + "=" * 80)
    print(df.drop(columns=["differs_from", "run_id"]).to_string(index=False))

    if args.no_register or len(results) < 2:
        print("\n[*] Registry step skipped.")
        return
    champ = max(results, key=lambda r: (r["val"]["roc_auc"], r["val"]["f1"], -r["val_loss"]))
    thr_path = threshold_figure(champ)
    champ, info = register(results, df, csv_path, chart_path, thr_path)
    print("\n" + "=" * 80 + "\nCHAMPION (chosen by VALIDATION ROC-AUC)\n" + "=" * 80)
    print(json.dumps(info, indent=2))
    print("\nFor the web app set (in .env):")
    print(f"  MODEL_PATH=champion_model.keras\n  MODEL_NAME={champ['cfg']['name']}\n  THRESHOLD={champ['thr']:.2f}")
    print(f"\nModels view: https://dagshub.com/{owner}/{repo}/models")


if __name__ == "__main__":
    main()
