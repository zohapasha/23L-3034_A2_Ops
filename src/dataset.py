import keras
import pandas as pd
import tensorflow as tf

from config import ROOT, SPLITS_DIR

AUTOTUNE = tf.data.AUTOTUNE


def load_image(image_bytes, size):
    # same function is used in the web app so preprocessing matches
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size))
    img.set_shape((size, size, 3))
    return img


def augmentation(seed=42):
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(0.05, seed=seed),
        keras.layers.RandomZoom(0.1, seed=seed),
        keras.layers.RandomContrast(0.1, seed=seed),
    ], name="augmentation")


def get_dataset(split, size=160, batch_size=32, augment=False, shuffle=False, seed=42):
    df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    paths = [str(ROOT / p) for p in df.path]
    labels = df.label.astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed)
    ds = ds.map(lambda p, y: (load_image(tf.io.read_file(p), size), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment:
        aug = augmentation(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import OUTPUT_DIR

    images, labels = next(iter(get_dataset("train", augment=True, shuffle=True, batch_size=16)))
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(tf.cast(tf.clip_by_value(images[i], 0, 255), tf.uint8))
        ax.set_title("ai" if labels[i] == 1 else "real")
        ax.axis("off")
    fig.suptitle("Augmented training batch")
    fig.tight_layout()
    (OUTPUT_DIR / "pipeline").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "pipeline" / "batch.png", dpi=110)
    print("batch shape:", images.shape)
    print("saved outputs/pipeline/batch.png")
