"""Input pipeline for VisionGuard: split lists and tf.data datasets."""
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
LABELS = {"Real": 0, "AI": 1}  # label 1 = AI-generated (positive class)
AUDIT_CSV = Path("reports/audit_table.csv")
SPLIT_DIR = Path("data/splits")


def make_splits(seed=SEED):
    """Stratified 70/15/15 split lists built from the audit table."""
    df = pd.read_csv(AUDIT_CSV)
    df = df[~df["corrupt"]]
    df = df.drop_duplicates("sha256", keep="first")  # keep one copy of exact duplicates
    df = df[["path", "class", "sha256"]].copy()
    df["label"] = df["class"].map(LABELS)

    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values("path").to_csv(SPLIT_DIR / f"{name}.csv", index=False)
    return train, val, test


def load_image(path, label, img_size=IMG_SIZE):
    """Read, decode (always 3 channels) and resize one image."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear", antialias=True)
    img = tf.cast(tf.round(img), tf.uint8)  # uint8 keeps the cache small
    return img, label


def build_augmenter(seed=SEED):
    """Mild random augmentation, used for the training split only."""
    L = tf.keras.layers
    return tf.keras.Sequential(
        [
            L.RandomFlip("horizontal", seed=seed),
            L.RandomRotation(0.04, fill_mode="reflect", seed=seed),  # about +/- 14 degrees
            L.RandomZoom(0.1, fill_mode="reflect", seed=seed),
            L.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
            L.RandomContrast(0.1, seed=seed),
        ],
        name="augmentation",
    )


def build_dataset(split, training, batch_size=BATCH_SIZE, img_size=IMG_SIZE, seed=SEED):
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    paths = df["path"].values
    labels = df["label"].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # few parallel calls on purpose: some images are up to ~60 megapixels
    ds = ds.map(lambda p, y: load_image(p, y, img_size), num_parallel_calls=4)
    ds = ds.cache()
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training:
        aug = build_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(tf.cast(x, tf.float32), training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    audit = pd.read_csv(AUDIT_CSV)
    tr, va, te = make_splits()
    all_ = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])
    table = pd.crosstab(all_["split"], all_["class"], margins=True)
    text = (
        f"Audited images                 : {len(audit)}\n"
        f"Used (no corrupt, no duplicates): {len(all_)}\n"
        f"Removed                        : {len(audit) - len(all_)}\n\n"
        f"{table.to_string()}\n\n"
        f"label 1 = AI-generated, label 0 = Real | seed = {SEED} | image size = {IMG_SIZE}\n"
        f"val element_spec: {build_dataset('val', training=False).element_spec}"
    )
    print(text)
    Path("reports/split_summary.txt").write_text(text, encoding="utf-8")