import pandas as pd
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, SPLITS_DIR

# tensorflow can only decode these
OK_FORMATS = ["JPEG", "PNG", "BMP", "GIF", "MPO"]

df = pd.read_csv(OUTPUT_DIR / "audit" / "audit_table.csv")
total = len(df)

# same image in both folders = we can't trust its label, drop it
per_hash = df.groupby("sha256")["class"].nunique()
conflicts = per_hash[per_hash > 1].index

df = df[df.readable & ~df.duplicate & ~df.sha256.isin(conflicts) & df.format.isin(OK_FORMATS)]
print(f"kept {len(df)} / {total} images")

df = df.assign(label=(df["class"] == "ai").astype(int))
df = df[["path", "label"]]

train, rest = train_test_split(df, test_size=0.3, stratify=df.label, random_state=42)
val, test = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=42)

SPLITS_DIR.mkdir(exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
    print(f"{name}: {len(part)}  (ai={part.label.sum()}, real={(part.label == 0).sum()})")