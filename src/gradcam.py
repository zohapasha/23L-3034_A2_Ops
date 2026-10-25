import argparse
import json

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from config import OUTPUT_DIR, ROOT, SPLITS_DIR  # noqa: E402
from dataset import load_image  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="efficientnet_default")
parser.add_argument("--n", type=int, default=4)
args = parser.parse_args()

run_dir = OUTPUT_DIR / "runs" / args.run
model = keras.models.load_model(run_dir / "best.keras")
thr = json.loads((run_dir / "results.json").read_text())["threshold"]
size = model.input_shape[1]

# split the model into: layers before the backbone, the backbone, and the head after it
layers = model.layers[1:]
base_idx = next((i for i, lyr in enumerate(layers) if isinstance(lyr, keras.Model)), None)
if base_idx is None:
    raise SystemExit("this script works on the transfer learning runs (mobilenet / efficientnet), not the cnn")
before, base, after = layers[:base_idx], layers[base_idx], layers[base_idx + 1:]


def heatmap(img):
    x = img[None]
    for lyr in before:
        x = lyr(x)
    with tf.GradientTape() as tape:
        feats = base(x, training=False)
        tape.watch(feats)
        out = feats
        for lyr in after:
            out = lyr(out, training=False)
        p = out[0, 0]
        # explain the class the model picked
        score = p if p >= thr else 1 - p
    grads = tape.gradient(score, feats)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (size, size)).numpy()[..., 0]
    return cam, float(p)


test = pd.read_csv(SPLITS_DIR / "test.csv")
images = [load_image(tf.io.read_file(str(ROOT / p)), size) for p in test.path]
probs = model.predict(tf.stack(images), verbose=0).ravel()
pred = (probs >= thr).astype(int)
right = np.where(pred == test.label.values)[0][:args.n]
wrong = np.where(pred != test.label.values)[0][:args.n]

fig, axes = plt.subplots(2, args.n, figsize=(4 * args.n, 8.5))
for row, (idx_list, title) in enumerate([(right, "correct"), (wrong, "wrong")]):
    for col in range(args.n):
        ax = axes[row, col]
        ax.axis("off")
        if col >= len(idx_list):
            continue
        i = idx_list[col]
        cam, p = heatmap(images[i])
        ax.imshow(tf.cast(images[i], tf.uint8))
        ax.imshow(cam, cmap="jet", alpha=0.4)
        true = "ai" if test.label[i] else "real"
        guess = "ai" if p >= thr else "real"
        ax.set_title(f"{title}: true={true}, pred={guess}\np(ai)={p:.2f}", fontsize=10)
fig.suptitle(f"Grad-CAM ({args.run}) - top: correct, bottom: wrong")
fig.tight_layout()
out = OUTPUT_DIR / "gradcam.png"
fig.savefig(out, dpi=110)
print("saved", out)
