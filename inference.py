import os
import io
import numpy as np
from PIL import Image
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow import keras

load_dotenv()

# Task 4.4: Read settings from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "champion_mobilenetv2_best.keras")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
MODEL_NAME = os.getenv("MODEL_NAME", "champion_mobilenetv2")

print("="*80)
print("INITIALIZING INFERENCE MODULE (TASK 4.1)")
print("="*80)
print(f"Loading champion model from: '{MODEL_PATH}'...")

# CRUCIAL (Sub-task 4.1): Load ONCE at module import / application start-up
# NEVER load inside the request handler
try:
    CHAMPION_MODEL = keras.models.load_model(MODEL_PATH)
    print(f"[+] Model '{MODEL_NAME}' successfully loaded into memory once at start-up.")
except Exception as e:
    print(f"[!] Warning loading model: {e}")
    CHAMPION_MODEL = None

def preprocess_image_bytes(image_bytes, target_size=(128, 128)):
    """
    Sub-task 4.1: Preprocesses an uploaded in-memory image exactly as in training:
    1. Reads byte buffer into PIL Image without writing to disk.
    2. Converts to 3-channel RGB (handles RGBA, Grayscale, CMYK).
    3. Resizes to (128, 128) using bilinear interpolation.
    4. Converts to float32 NumPy tensor.
    5. Adds batch dimension: shape (1, 128, 128, 3).
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify() # Check integrity
        # Re-open after verify
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Image decode error: File is corrupted or unreadable ({e})")

    pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(pil_img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def predict(image_bytes):
    """
    Runs model inference and returns verdict and confidence percentage.
    """
    if CHAMPION_MODEL is None:
        raise RuntimeError("Model is not initialized.")

    img_tensor = preprocess_image_bytes(image_bytes)
    raw_prob = float(CHAMPION_MODEL.predict(img_tensor, verbose=0)[0][0])

    if raw_prob >= THRESHOLD:
        verdict = "AI-Generated Image"
        confidence = raw_prob * 100.0
    else:
        verdict = "Real Photography"
        confidence = (1.0 - raw_prob) * 100.0

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "raw_score": round(raw_prob, 4),
        "threshold": THRESHOLD,
        "model_name": MODEL_NAME
    }

if __name__ == "__main__":
    # Test console output for two known images (Evidence for Sub-task 4.1)
    print("\n" + "="*80)
    print("SUB-TASK 4.1 EVIDENCE: CONSOLE OUTPUT FOR TWO KNOWN IMAGES")
    print("="*80)
    
    ai_files = [f for f in os.listdir("./dataset/AI") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    real_files = [f for f in os.listdir("./dataset/Real") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if ai_files:
        test_ai_path = os.path.join("./dataset/AI", ai_files[0])
        with open(test_ai_path, "rb") as f:
            res_ai = predict(f.read())
        print(f"[Known AI Image: {os.path.basename(test_ai_path)}]")
        print(f"  -> Verdict   : {res_ai['verdict']}")
        print(f"  -> Confidence: {res_ai['confidence']}% (Raw Score: {res_ai['raw_score']})")

    if real_files:
        test_real_path = os.path.join("./dataset/Real", real_files[0])
        with open(test_real_path, "rb") as f:
            res_real = predict(f.read())
        print(f"\n[Known Real Image: {os.path.basename(test_real_path)}]")
        print(f"  -> Verdict   : {res_real['verdict']}")
        print(f"  -> Confidence: {res_real['confidence']}% (Raw Score: {res_real['raw_score']})")
    print("="*80)
