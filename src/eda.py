import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from config import CLASSES, OUTPUT_DIR, ROOT  # noqa: E402

out_dir = OUTPUT_DIR / "eda"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / "audit" / "audit_table.csv")
df = df[df.readable & ~df.duplicate]

# 8 random images from each class
n = 8
fig, axes = plt.subplots(2, n, figsize=(16, 4.6))
for r, label in enumerate(CLASSES):
    subset = df[df["class"] == label]
    sample = subset.sample(min(n, len(subset)), random_state=1)
    for c, row in enumerate(sample.itertuples()):
        ax = axes[r, c]
        with Image.open(ROOT / row.path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{label} {int(row.width)}x{int(row.height)}", fontsize=8)
        ax.axis("off")
fig.suptitle("Sample images (top: real, bottom: AI)")
fig.tight_layout()
fig.savefig(out_dir / "samples.png", dpi=120)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col, title in zip(axes, ["width", "height", "size_kb"], ["Width (px)", "Height (px)", "File size (KB)"]):
    # file sizes go from ~10 KB to ~10 MB, so use log-spaced bins for that one
    bins = np.logspace(np.log10(df[col].min()), np.log10(df[col].max()), 30) if col == "size_kb" else 30
    for label in CLASSES:
        ax.hist(df.loc[df["class"] == label, col], bins=bins, alpha=0.6, label=label)
    ax.set_xlabel(title)
    ax.set_ylabel("count")
    ax.legend()
axes[2].set_xscale("log")
fig.suptitle("Image width, height and file size by class")
fig.tight_layout()
fig.savefig(out_dir / "histograms.png", dpi=120)

print("saved", out_dir / "samples.png", "and", out_dir / "histograms.png")