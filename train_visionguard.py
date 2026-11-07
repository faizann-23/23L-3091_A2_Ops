import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# 1. Training Discipline: Fix Seeds (Task 2.4)
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def prepare_splits(audit_csv="dataset_audit.csv"):
    """Creates stratified train (70%), val (15%), test (15%) splits."""
    df = pd.read_csv(audit_csv)
    # Deduplicate exact duplicate files discovered during audit
    df = df[~df["is_duplicate"]].reset_index(drop=True)
    
    # AI = 1, Real = 0
    df["label"] = df["class"].apply(lambda c: 1 if c == "AI" else 0)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
    )

    train_df.to_csv("train_split.csv", index=False)
    val_df.to_csv("val_split.csv", index=False)
    test_df.to_csv("test_split.csv", index=False)

    print(f"[+] Splits created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

# Task 2.1: Input pipeline with tf.data
def parse_image(filepath, label):
    img_bytes = tf.io.read_file(filepath)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE) # uniform resizing
    return img, label

def get_data_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

def build_dataset(df, is_training=False):
    filepaths = df["filepath"].values
    labels = df["label"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
        aug = get_data_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def save_augmented_batch_sample(train_ds):
    """Sub-task 2.1 evidence: figure of one augmented batch."""
    for images, labels in train_ds.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            lbl = "AI-Gen" if labels[i].numpy() == 1 else "Real"
            plt.title(f"Augmented: {lbl}")
            plt.axis("off")
        plt.suptitle("Task 2.1: Augmented Training Batch Sample", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("task2_augmented_batch.png", dpi=200)
        plt.close()
        print("[+] Saved augmented batch sample to 'task2_augmented_batch.png'.")
        break

# Task 2.2: Baseline CNN
def build_baseline_cnn():
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # Rescale [0, 255] -> [0, 1]
    x = layers.Rescaling(1./255)(inputs)
    
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model

# Task 2.3: Transfer Learning (MobileNetV2)
def build_transfer_learning_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # MobileNetV2 expects preprocessing in [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_transfer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model, base_model

def get_callbacks(model_name):
    """Task 2.4: Training discipline callbacks."""
    checkpoint_path = f"{model_name}_best.keras"
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True)
    ], checkpoint_path

def evaluate_and_plot(model, test_ds, test_df, model_name="model"):
    """Task 2.5 & 2.6: Comprehensive Evaluation & Error Analysis."""
    y_true = test_df["label"].values
    
    # Predict probabilities
    y_pred_prob = model.predict(test_ds).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_prob)

    print("\n" + "="*60)
    print(f"EVALUATION RESULTS FOR: {model_name}")
    print("="*60)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("="*60)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "AI"], yticklabels=["Real", "AI"])
    plt.title(f"{model_name}: Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#2980b9", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.title(f"{model_name}: ROC Curve", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_name}_roc_curve.png", dpi=200)
    plt.close()

    # 3. Error Analysis: 10 Misclassified Images (Task 2.6)
    test_df_eval = test_df.copy()
    test_df_eval["pred_prob"] = y_pred_prob
    test_df_eval["pred_label"] = y_pred
    misclassified = test_df_eval[test_df_eval["label"] != test_df_eval["pred_label"]].head(10)
    
    if len(misclassified) > 0:
        n_mis = len(misclassified)
        cols = min(5, n_mis)
        rows = (n_mis + cols - 1) // cols
        plt.figure(figsize=(4 * cols, 4 * rows))
        for idx, (_, row) in enumerate(misclassified.iterrows()):
            ax = plt.subplot(rows, cols, idx + 1)
            img = Image.open(row["filepath"])
            ax.imshow(img)
            true_txt = "AI" if row["label"] == 1 else "Real"
            pred_txt = "AI" if row["pred_label"] == 1 else "Real"
            conf = row["pred_prob"] if row["pred_label"] == 1 else (1.0 - row["pred_prob"])
            ax.set_title(f"True: {true_txt} | Pred: {pred_txt}\nConf: {conf*100:.1f}%", fontsize=9)
            ax.axis("off")
        plt.suptitle(f"{model_name}: Error Analysis (Misclassified Samples)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{model_name}_misclassified_grid.png", dpi=200)
        plt.close()
        print(f"[+] Saved error analysis figure to '{model_name}_misclassified_grid.png'.")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc)
    }

def run_all_training():
    train_df, val_df, test_df = prepare_splits()

    train_ds = build_dataset(train_df, is_training=True)
    val_ds = build_dataset(val_df, is_training=False)
    test_ds = build_dataset(test_df, is_training=False)

    save_augmented_batch_sample(train_ds)

    # Model 1: Baseline CNN
    print("\n" + "="*80)
    print("TRAINING MODEL 1: BASELINE CNN")
    print("="*80)
    baseline_model = build_baseline_cnn()
    baseline_model.summary()
    callbacks, baseline_path = get_callbacks("baseline_cnn")
    history_baseline = baseline_model.fit(
        train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks, verbose=1
    )
    baseline_metrics = evaluate_and_plot(baseline_model, test_ds, test_df, "baseline_cnn")

    # Model 2: Transfer Learning (MobileNetV2)
    print("\n" + "="*80)
    print("TRAINING MODEL 2: TRANSFER LEARNING (MOBILENETV2)")
    print("="*80)
    transfer_model, base_mobilenet = build_transfer_learning_model()
    
    # Stage 1: Frozen base
    callbacks_tl, tl_path = get_callbacks("mobilenetv2_stage1")
    print("\n[*] Stage 1: Training classifier head with frozen MobileNetV2 base...")
    h1 = transfer_model.fit(
        train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks_tl, verbose=1
    )

    # Stage 2: Fine-tuning top layers
    print("\n[*] Stage 2: Unfreezing top 25 layers of MobileNetV2 with low learning rate...")
    base_mobilenet.trainable = True
    for layer in base_mobilenet.layers[:-25]:
        layer.trainable = False

    transfer_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    callbacks_ft, champion_path = get_callbacks("champion_mobilenetv2")
    h2 = transfer_model.fit(
        train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks_ft, verbose=1
    )

    # Plot Transfer Learning Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    plt.plot(acc, label="Training Accuracy", color="#27ae60")
    plt.plot(val_acc, label="Validation Accuracy", color="#2980b9")
    plt.axvline(x=len(h1.history["accuracy"])-1, color="red", linestyle="--", label="Fine-Tuning Start")
    plt.title("MobileNetV2: Accuracy Across Stages", fontweight="bold")
    plt.legend()

    plt.subplot(1, 2, 2)
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]
    plt.plot(loss, label="Training Loss", color="#e74c3c")
    plt.plot(val_loss, label="Validation Loss", color="#8e44ad")
    plt.axvline(x=len(h1.history["loss"])-1, color="red", linestyle="--", label="Fine-Tuning Start")
    plt.title("MobileNetV2: Loss Across Stages", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task2_transfer_learning_curves.png", dpi=200)
    plt.close()
    print("[+] Saved transfer learning curves to 'task2_transfer_learning_curves.png'.")

    transfer_metrics = evaluate_and_plot(transfer_model, test_ds, test_df, "champion_mobilenetv2")

    # Comparison Table
    comparison = pd.DataFrame([
        {"Model": "Baseline CNN", **baseline_metrics},
        {"Model": "Transfer Learning (MobileNetV2)", **transfer_metrics}
    ])
    print("\n" + "="*80)
    print("SUB-TASK 2.5: MODEL COMPARISON TABLE (EVALUATION ON TEST SET)")
    print("="*80)
    print(comparison.to_string(index=False))
    print("="*80)
    comparison.to_csv("task2_model_comparison.csv", index=False)

if __name__ == "__main__":
    run_all_training()
