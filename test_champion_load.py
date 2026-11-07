"""Sub-task 3.6: load the registered model BY ALIAS (@champion) and classify one image.

The model is fetched through the registry URI  models:/VisionGuard_Classifier@champion,
so whichever version holds the alias is the one that is used - no file name is hard-coded.
Credentials come from environment variables / .env (never from this file).

Usage:
    python test_champion_load.py [path/to/image.jpg]
"""
import glob
import json
import os
import sys

import mlflow
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from tensorflow import keras

load_dotenv()

MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "visionguard-detector")
ALIAS = "champion"

repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "faizann-23")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "23L-3091_A2_Ops")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI",
                         f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
if "dagshub.com" in tracking_uri:
    token = os.getenv("DAGSHUB_USER_TOKEN")
    if not token:
        sys.exit("DAGSHUB_USER_TOKEN is not set. Put it in .env (see .env.example).")
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", repo_owner)
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
mlflow.set_tracking_uri(tracking_uri)


def load_champion():
    """Resolve the alias, download that version's artifacts, return (keras_model, version)."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    print(f"[+] Alias '@{ALIAS}' -> version {mv.version} (run {mv.run_id})")

    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}@{ALIAS}")
    keras_files = glob.glob(os.path.join(local_dir, "**", "*.keras"), recursive=True)
    if not keras_files:
        sys.exit(f"No .keras file found inside the registered artifacts at {local_dir}")
    print(f"[+] Downloaded registry artifact: {keras_files[0]}")
    return keras.models.load_model(keras_files[0]), mv.version


def champion_threshold():
    """Decision threshold chosen on the validation set (written by log_visionguard_experiments.py)."""
    if os.path.exists("champion_info.json"):
        with open("champion_info.json") as f:
            return float(json.load(f)["threshold"])
    return float(os.getenv("THRESHOLD", "0.5"))


def load_image(path, size):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, size)  # bilinear, same size the model was trained with
    return tf.expand_dims(img, axis=0)


def main(image_path=None):
    print("=" * 80)
    print("SUB-TASK 3.6: LOAD MODEL BY ALIAS (@champion) & RUN INFERENCE")
    print("=" * 80)
    print(f"Tracking URI : {tracking_uri}")
    print(f"Target Model : models:/{MODEL_NAME}@{ALIAS}")

    model, version = load_champion()

    if image_path and os.path.exists(image_path):
        ground_truth = "Unknown"
    else:
        row = pd.read_csv("test_split.csv").iloc[0]
        image_path = row["filepath"]
        ground_truth = "AI-Generated" if row["label"] == 1 else "Real Photography"

    print(f"\n[+] Input image   : {image_path}")
    print(f"[+] Ground truth  : {ground_truth}")

    size = tuple(model.input_shape[1:3])
    threshold = champion_threshold()
    prob = float(model.predict(load_image(image_path, size), verbose=0)[0][0])
    is_ai = prob >= threshold
    verdict = "AI-Generated Image" if is_ai else "Real Photography"
    confidence = (prob if is_ai else 1.0 - prob) * 100

    print("\n" + "=" * 80)
    print("VISIONGUARD CLASSIFICATION RESULT")
    print("=" * 80)
    print(f"Model version      : {MODEL_NAME} v{version} (@{ALIAS})")
    print(f"Prediction Verdict : {verdict.upper()}")
    print(f"Confidence Score   : {confidence:.2f}%")
    print(f"Raw Sigmoid Output : {prob:.4f} (threshold {threshold:.2f}, input {size[0]}x{size[1]})")
    if ground_truth != "Unknown":
        ok = (is_ai and ground_truth == "AI-Generated") or (not is_ai and ground_truth == "Real Photography")
        print(f"Match Ground Truth : {'YES (CORRECT)' if ok else 'NO'}")
    print("=" * 80)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
