"""MobileNetV2 transfer-learning model (two-stage: frozen, then last block unfrozen)."""
import keras
import numpy as np
from keras import layers
from keras.applications import MobileNetV2

LAST_BLOCK_PREFIXES = ("block_16", "Conv_1")  # last inverted-residual block + final 1x1 conv


def build_mobilenet(img_size=224, dropout=0.3, weights="imagenet"):
    base = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights=weights)
    base.trainable = False  # stage 1: everything in the backbone is frozen

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    # same as keras.applications.mobilenet_v2.preprocess_input: [0, 255] -> [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    x = base(x, training=False)  # keeps BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    return keras.Model(inputs, outputs, name="visionguard_mobilenetv2"), base


def unfreeze_last_block(base):
    """Stage 2: only the last block is trainable; BatchNorm layers stay frozen."""
    base.trainable = True
    for layer in base.layers:
        is_bn = isinstance(layer, layers.BatchNormalization)
        layer.trainable = layer.name.startswith(LAST_BLOCK_PREFIXES) and not is_bn


def describe_trainable(model, base, stage):
    n_tr = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    lines = [f"=== {stage} ===",
             f"trainable parameters: {n_tr:,} of {model.count_params():,}"]
    in_base = [l.name for l in base.layers if l.trainable and l.weights]
    lines.append(f"trainable layers inside the backbone: {len(in_base)}")
    lines += [f"    {n}" for n in in_base]
    head = [l.name for l in model.layers if l is not base and l.trainable and l.weights]
    lines.append("trainable layers in the classifier head: " + (", ".join(head) or "none"))
    return "\n".join(lines)