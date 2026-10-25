import re

import keras
from keras import layers


def simple_cnn(size=160, dropout=0.4):
    inputs = keras.Input((size, size, 3))
    x = layers.Rescaling(1 / 255)(inputs)
    for i, filters in enumerate([32, 64, 128], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv4")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs, outputs, name="cnn")


def pretrained(backbone="mobilenetv2", size=160, dropout=0.3, weights="imagenet"):
    if backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=(size, size, 3))
        # mobilenetv2 was trained on pixels in [-1, 1]
        preprocess = layers.Rescaling(1 / 127.5, offset=-1)
    else:
        base = keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=(size, size, 3))
        # efficientnet does its own scaling and expects 0-255
        preprocess = layers.Identity()
    base.trainable = False

    inputs = keras.Input((size, size, 3))
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs, outputs, name=backbone), base


def unfreeze_top(base, n_layers=None):
    """n_layers=None -> unfreeze the last block only. BatchNorm stays frozen."""
    base.trainable = True
    names = [layer.name for layer in base.layers]
    if n_layers is None:
        blocks = [m.group(1) for m in (re.match(r"(block_?\d+[a-z]?)_", n) for n in names) if m]
        start = next(i for i, n in enumerate(names) if n.startswith(blocks[-1] + "_"))
    else:
        start = max(0, len(names) - n_layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    return [layer.name for layer in base.layers if layer.trainable and layer.weights]


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    return model
