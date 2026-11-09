"""Backbone selection and partial unfreezing for the MLflow experiments."""
import keras
import numpy as np
from keras import layers
from keras.applications import EfficientNetB0

from transfer import build_mobilenet


def build_model(backbone, img_size=224, dropout=0.3):
    """Returns (model, base). Model input: RGB pixels in [0, 255]; output: P(image is AI-generated)."""
    if backbone == "mobilenetv2":
        return build_mobilenet(img_size=img_size, dropout=dropout)   # rescales to [-1, 1] inside
    if backbone == "efficientnetb0":
        base = EfficientNetB0(include_top=False, weights="imagenet",
                              input_shape=(img_size, img_size, 3))   # has its own preprocessing inside
        base.trainable = False
        inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dropout(dropout, name="dropout")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
        return keras.Model(inputs, outputs, name="visionguard_efficientnetb0"), base
    raise ValueError(f"unknown backbone: {backbone}")


def set_unfrozen(base, n):
    """Freeze the whole backbone, then unfreeze the last n layers (BatchNorm always stays frozen)."""
    base.trainable = n > 0
    if n > 0:
        top = {id(l) for l in base.layers[-n:]}
        for layer in base.layers:
            layer.trainable = (id(layer) in top) and not isinstance(layer, layers.BatchNormalization)


def count_trainable(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)