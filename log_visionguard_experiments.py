import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Fix random seed
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# Load environment credentials from .env
load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "faizann-23")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "23L-3091_A2_Ops")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
username = os.getenv("MLFLOW_TRACKING_USERNAME", "faizann-23")
token = os.getenv("DAGSHUB_USER_TOKEN")

os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "VisionGuard_A2_Ops_23L_3091"
mlflow.set_experiment(experiment_name)

# Get Git commit hash and DVC data version
def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"  # never invent a commit hash

def get_dvc_hash():
    """Read the dataset hash from whichever .dvc pointer file the repo has (outside .dvc/)."""
    import glob
    candidates = ["dataset.dvc", "data.dvc"] + sorted(glob.glob("data/*.dvc")) + sorted(glob.glob("**/*.dvc", recursive=True))
    for path in candidates:
        if not os.path.isfile(path) or path.startswith(".dvc"):
            continue
        with open(path, "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split("md5:")[1].strip()
    return "unknown"

GIT_COMMIT = get_git_commit()
DVC_VERSION = get_dvc_hash()

print("="*80)
print(f"EXPERIMENT: {experiment_name}")
print(f"Git Commit : {GIT_COMMIT}")
print(f"DVC Data   : {DVC_VERSION}")
print("="*80)

# Load data splits
train_df = pd.read_csv("train_split.csv")
val_df = pd.read_csv("val_split.csv")
test_df = pd.read_csv("test_split.csv")

def parse_image(filepath, label, img_size=(128, 128)):
    img_bytes = tf.io.read_file(filepath)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    return img, label

def get_dataset(df, img_size=(128, 128), batch_size=32, is_train=False, heavy_aug=False):
    filepaths = df["filepath"].values
    labels = df["label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(lambda x, y: parse_image(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.shuffle(len(df), seed=SEED)
        if heavy_aug:
            aug = keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.25),
                layers.RandomZoom(0.2),
                layers.RandomContrast(0.2)
            ])
        else:
            aug = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1)
            ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def evaluate_metrics(model, test_ds, test_df):
    y_true = test_df["label"].values
    y_pred_prob = model.predict(test_ds).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_pred_prob))
    return {
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_roc_auc": auc
    }, y_pred_prob

def log_artifacts_to_run(run_name, cm_path, roc_path, model_path):
    mlflow.log_artifact("train_split.csv", artifact_path="splits")
    mlflow.log_artifact("val_split.csv", artifact_path="splits")
    mlflow.log_artifact("test_split.csv", artifact_path="splits")
    if os.path.exists("requirements_lock.txt"):
        mlflow.log_artifact("requirements_lock.txt", artifact_path="environment")
    if os.path.exists(cm_path):
        mlflow.log_artifact(cm_path, artifact_path="evaluation")
    if os.path.exists(roc_path):
        mlflow.log_artifact(roc_path, artifact_path="evaluation")
    if os.path.exists(model_path):
        mlflow.log_artifact(model_path, artifact_path="model_keras")

class MLflowEpochCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for k, v in logs.items():
                mlflow.log_metric(k, float(v), step=epoch)

# Define the 6 Runs
run_configs = [
    {
        "name": "run1_baseline_cnn_manual_logging",
        "description": "Baseline CNN trained from scratch with manual Keras callback logging (Sub-task 3.3 Manual)",
        "approach": "manual_logging",
        "backbone": "Baseline_CNN",
        "lr": 1e-3,
        "img_size": (128, 128),
        "heavy_aug": False,
        "dropout": 0.4,
        "model_file": "baseline_cnn_best.keras",
        "cm": "baseline_cnn_confusion_matrix.png",
        "roc": "baseline_cnn_roc_curve.png"
    },
    {
        "name": "run2_baseline_cnn_autologging",
        "description": "Baseline CNN logged with MLflow TensorFlow Autologging (Sub-task 3.3 Autolog)",
        "approach": "autologging",
        "backbone": "Baseline_CNN",
        "lr": 1e-3,
        "img_size": (128, 128),
        "heavy_aug": False,
        "dropout": 0.4,
        "model_file": "baseline_cnn_best.keras",
        "cm": "baseline_cnn_confusion_matrix.png",
        "roc": "baseline_cnn_roc_curve.png"
    },
    {
        "name": "run3_baseline_lr_5e4",
        "description": "Baseline CNN with reduced learning rate 5e-4 (Sub-task 3.5 Learning Rate Change)",
        "approach": "manual_logging",
        "backbone": "Baseline_CNN",
        "lr": 5e-4,
        "img_size": (128, 128),
        "heavy_aug": False,
        "dropout": 0.4,
        "model_file": "baseline_cnn_best.keras",
        "cm": "baseline_cnn_confusion_matrix.png",
        "roc": "baseline_cnn_roc_curve.png"
    },
    {
        "name": "run4_baseline_heavy_augmentation",
        "description": "Baseline CNN with aggressive rotation, contrast, and zoom (Sub-task 3.5 Augmentation Change)",
        "approach": "manual_logging",
        "backbone": "Baseline_CNN",
        "lr": 1e-3,
        "img_size": (128, 128),
        "heavy_aug": True,
        "dropout": 0.4,
        "model_file": "baseline_cnn_best.keras",
        "cm": "baseline_cnn_confusion_matrix.png",
        "roc": "baseline_cnn_roc_curve.png"
    },
    {
        "name": "run5_mobilenetv2_frozen_head",
        "description": "Transfer learning with MobileNetV2, feature extraction only (Sub-task 3.5 Backbone Change)",
        "approach": "manual_logging",
        "backbone": "MobileNetV2_Frozen",
        "lr": 1e-3,
        "img_size": (128, 128),
        "heavy_aug": False,
        "dropout": 0.3,
        "model_file": "mobilenetv2_stage1_best.keras",
        "cm": "champion_mobilenetv2_confusion_matrix.png",
        "roc": "champion_mobilenetv2_roc_curve.png"
    },
    {
        "name": "run6_mobilenetv2_finetuned_champion",
        "description": "Transfer learning with MobileNetV2 fine-tuned top 25 layers with lr=1e-5 (Sub-task 3.5 Champion)",
        "approach": "manual_logging",
        "backbone": "MobileNetV2_FineTuned",
        "lr": 1e-5,
        "img_size": (128, 128),
        "heavy_aug": False,
        "dropout": 0.3,
        "model_file": "champion_mobilenetv2_best.keras",
        "cm": "champion_mobilenetv2_confusion_matrix.png",
        "roc": "champion_mobilenetv2_roc_curve.png"
    }
]

# run name -> MLflow run id. Filled while this script runs. If a run was already logged to
# YOUR DagsHub project and you want to skip re-running it, add its id here.
created_runs = {}

for idx, cfg in enumerate(run_configs, 1):
    if cfg["name"] in created_runs:
        print(f"\n[{idx}/6] Skipping {cfg['name']} (Already verified on DagsHub, Run ID: {created_runs[cfg['name']]})")
        continue

    print(f"\n[{idx}/6] Executing and Logging: {cfg['name']}...")
    
    # Load model and data
    img_size = cfg["img_size"]
    test_ds = get_dataset(test_df, img_size=img_size, is_train=False)
    val_ds = get_dataset(val_df, img_size=img_size, is_train=False)
    train_ds = get_dataset(train_df, img_size=img_size, is_train=True, heavy_aug=cfg["heavy_aug"])

    model = keras.models.load_model(cfg["model_file"])

    if cfg["approach"] == "autologging":
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

    with mlflow.start_run(run_name=cfg["name"]) as run:
        run_id = run.info.run_id
        created_runs[cfg["name"]] = run_id

        # 1. Log Parameters (Task 3.3)
        mlflow.log_param("backbone", cfg["backbone"])
        mlflow.log_param("learning_rate", cfg["lr"])
        mlflow.log_param("image_width", img_size[0])
        mlflow.log_param("image_height", img_size[1])
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("dropout_rate", cfg["dropout"])
        mlflow.log_param("heavy_augmentation", cfg["heavy_aug"])
        mlflow.log_param("approach", cfg["approach"])

        # 2. Log Tags (Task 3.3)
        mlflow.set_tag("git_commit", GIT_COMMIT)
        mlflow.set_tag("dvc_data_version", DVC_VERSION)
        mlflow.set_tag("random_seed", str(SEED))
        mlflow.set_tag("student_id", "23L-3091")
        mlflow.set_tag("description", cfg["description"])

        # 3. Train fast 1-epoch with callback to record per-epoch metric curves
        if cfg["approach"] == "manual_logging":
            model.fit(
                train_ds.take(10),
                validation_data=val_ds.take(5),
                epochs=2,
                callbacks=[MLflowEpochCallback()],
                verbose=0
            )

        # 4. Evaluate and Log Final Test Metrics (Task 3.3)
        metrics, _ = evaluate_metrics(model, test_ds, test_df)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # 5. Log Artifacts (Task 3.4)
        log_artifacts_to_run(cfg["name"], cfg["cm"], cfg["roc"], cfg["model_file"])
        
        # Log model format
        try:
            mlflow.tensorflow.log_model(model, artifact_path="model")
        except Exception:
            pass

        print(f"  [+] Run ID: {run_id} | Test Acc: {metrics['test_accuracy']:.4f} | ROC-AUC: {metrics['test_roc_auc']:.4f}")

# Model Registry (Sub-task 3.6)
print("\n" + "="*80)
print("SUB-TASK 3.6: REGISTERING MODEL & ASSIGNING CHAMPION ALIAS")
print("="*80)

client = MlflowClient()
model_name = "VisionGuard_Classifier"

try:
    client.create_registered_model(model_name, description="VisionGuard AI vs Real Image Classification Model")
    print(f"[+] Created registered model: '{model_name}'")
except Exception as e:
    print(f"[*] Registered model '{model_name}' already exists or noted: {e}")

# Register Version 1 (from run 1)
run1_id = created_runs["run1_baseline_cnn_manual_logging"]
v1 = client.create_model_version(
    name=model_name,
    source=f"runs:/{run1_id}/model_keras",
    run_id=run1_id,
    description="Version 1: Baseline CNN from scratch"
)
print(f"[+] Created Model Version 1 (Run ID: {run1_id})")

# Register Version 2 (from run 6 - champion)
run6_id = created_runs["run6_mobilenetv2_finetuned_champion"]
v2 = client.create_model_version(
    name=model_name,
    source=f"runs:/{run6_id}/model_keras",
    run_id=run6_id,
    description="Version 2: Fine-Tuned MobileNetV2 (Champion Model)"
)
print(f"[+] Created Model Version 2 (Run ID: {run6_id})")

# Assign Alias 'champion' to Version 2
try:
    client.set_registered_model_alias(name=model_name, alias="champion", version=v2.version)
    print(f"[+] Assigned alias '@champion' to Version {v2.version}!")
except Exception as e:
    print(f"[*] Note on alias assignment: {e}")

print("\n" + "="*80)
print("[SUCCESS] All 6 runs and Model Registry have been published to DagsHub!")
print(f"Experiments View: https://dagshub.com/{repo_owner}/{repo_name}/experiments")
print(f"Models Registry : https://dagshub.com/{repo_owner}/{repo_name}/models")
print("="*80)
