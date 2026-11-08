from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from input_pipeline import SEED, SPLIT_DIR, build_augmenter, build_dataset, load_image

tf.keras.utils.set_random_seed(SEED)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# 1) real batches: shape, value range, and "augmentation only on train"
train_ds = build_dataset("train", training=True)
val_ds = build_dataset("val", training=False)
xb, yb = next(iter(train_ds))
print("train batch:", xb.shape, xb.dtype,
      "| min/max:", float(tf.reduce_min(xb)), float(tf.reduce_max(xb)),
      "| share of AI labels:", round(float(tf.reduce_mean(yb)), 2))
xv1, _ = next(iter(val_ds))
xv2, _ = next(iter(val_ds))
xt1, _ = next(iter(train_ds))
xt2, _ = next(iter(train_ds))
print("val batch identical on two passes  :", bool(np.allclose(xv1, xv2)))
print("train batch identical on two passes:", bool(np.allclose(xt1, xt2)))

# 2) figure: 8 resized originals (top) and their augmented versions (bottom)
df = pd.read_csv(SPLIT_DIR / "train.csv")
sample = pd.concat([df[df.label == 1].sample(4, random_state=SEED),
                    df[df.label == 0].sample(4, random_state=SEED)])
imgs = np.stack([load_image(p, float(l))[0].numpy()
                 for p, l in zip(sample.path, sample.label)]).astype("float32")
auged = build_augmenter(SEED)(imgs, training=True).numpy()

fig, axes = plt.subplots(2, 8, figsize=(20, 5.8))
for i in range(8):
    axes[0, i].imshow(imgs[i].astype("uint8"))
    axes[0, i].set_title("AI" if sample.label.iloc[i] == 1 else "Real")
    axes[1, i].imshow(np.clip(auged[i], 0, 255).astype("uint8"))
    axes[0, i].axis("off")
    axes[1, i].axis("off")
fig.suptitle("Top: resized originals (224x224)  |  Bottom: after random augmentation (training split only)")
fig.tight_layout()
fig.savefig("reports/figures/augmented_batch.png", dpi=150)
print("Saved reports/figures/augmented_batch.png")