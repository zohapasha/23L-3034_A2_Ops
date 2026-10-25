import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import OUTPUT_DIR, PROJECT, start_mlflow  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--local", action="store_true")
args = parser.parse_args()
mlflow = start_mlflow(local=args.local)

runs = mlflow.search_runs(experiment_names=[PROJECT])
runs = runs[runs["metrics.test_f1"].notna()]  # skips the finetune child runs

columns = {
    "tags.mlflow.runName": "run",
    "params.cfg_model": "model",
    "params.cfg_backbone": "backbone",
    "params.cfg_lr": "lr",
    "params.cfg_img_size": "img",
    "params.cfg_no_augment": "no_aug",
    "params.cfg_unfreeze": "unfreeze",
    "metrics.val_auc": "val_auc",
    "metrics.best_val_loss": "val_loss",
    "metrics.test_accuracy": "test_acc",
    "metrics.test_precision": "test_prec",
    "metrics.test_recall": "test_rec",
    "metrics.test_f1": "test_f1",
    "metrics.test_auc": "test_auc",
    "metrics.threshold": "thr",
    "run_id": "run_id",
}
table = runs[[c for c in columns if c in runs]].rename(columns=columns)
table = table.sort_values(["val_auc", "val_loss"], ascending=[False, True]).round(4)

out_dir = OUTPUT_DIR / "comparison"
out_dir.mkdir(parents=True, exist_ok=True)
table.to_csv(out_dir / "runs.csv", index=False)
print(table.drop(columns="run_id").to_string(index=False))
print("\nbest run (val auc):", table.iloc[0]["run"])

fig, ax = plt.subplots(figsize=(11, 4.5))
x = range(len(table))
ax.bar([i - 0.2 for i in x], table.val_auc, width=0.4, label="val AUC")
ax.bar([i + 0.2 for i in x], table.test_f1, width=0.4, label="test F1")
ax.set_xticks(list(x), table.run, rotation=30, ha="right")
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("Run comparison")
fig.tight_layout()
fig.savefig(out_dir / "runs.png", dpi=110)
print("saved", out_dir / "runs.csv")
