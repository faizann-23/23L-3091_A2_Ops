"""Model definitions for VisionGuard."""
import keras
from keras import layers

IMG_SIZE = 224


def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    return x


def build_cnn(img_size=IMG_SIZE, filters=(16, 32, 64, 128), dropout=0.5):
    """Small CNN from scratch. Input: RGB pixels in [0, 255]. Output: P(image is AI-generated)."""
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for i, f in enumerate(filters, 1):
        x = conv_block(x, f, f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    return keras.Model(inputs, outputs, name="visionguard_cnn")


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model