import hashlib

import pandas as pd
from PIL import Image

from config import CLASSES, DATA_DIR, OUTPUT_DIR, ROOT

out_dir = OUTPUT_DIR / "audit"
out_dir.mkdir(parents=True, exist_ok=True)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def open_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() leaves the file unusable so open it again and decode
        with Image.open(path) as img:
            img.load()
            return True, img.width, img.height, img.mode, img.format, ""
    except Exception as e:
        return False, None, None, None, None, str(e)[:100]


rows = []
for label in CLASSES:
    folder = DATA_DIR / label
    if not folder.exists():
        raise SystemExit(f"folder not found: {folder}")
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        ok, w, h, mode, fmt, err = open_image(path)
        rows.append({
            "path": path.relative_to(ROOT).as_posix(),
            "class": label,
            "size_kb": round(path.stat().st_size / 1024, 1),
            "width": w,
            "height": h,
            "mode": mode,
            "format": fmt,
            "readable": ok,
            "error": err,
            "sha256": file_hash(path),
        })

df = pd.DataFrame(rows)
df["duplicate"] = df.duplicated("sha256")
df.to_csv(out_dir / "audit_table.csv", index=False)

good = df[df.readable]
counts = df.groupby("class").agg(
    files=("path", "count"),
    readable=("readable", "sum"),
    corrupt=("readable", lambda s: (~s).sum()),
    duplicates=("duplicate", "sum"),
)
sizes = good.groupby("class")[["width", "height", "size_kb"]].agg(["min", "median", "max"])
formats = good.groupby(["class", "format", "mode"]).size().reset_index(name="count")
both = df[df.duplicated("sha256", keep=False)].groupby("sha256")["class"].nunique()

counts.to_csv(out_dir / "counts.csv")
sizes.to_csv(out_dir / "sizes.csv")
formats.to_csv(out_dir / "formats.csv", index=False)

pd.set_option("display.width", 160)
print("\nFiles per class\n", counts)
print("\nWidth / height / size (KB)\n", sizes)
print("\nFormats and colour modes\n", formats.to_string(index=False))
print("\nCorrupt files:", int((~df.readable).sum()))
print("Duplicates:", int(df.duplicate.sum()))
print("Duplicates found in both classes:", int((both > 1).sum()))
if (~df.readable).any():
    print("\n", df.loc[~df.readable, ["path", "error"]].to_string(index=False))
print("\nsaved", out_dir / "audit_table.csv")