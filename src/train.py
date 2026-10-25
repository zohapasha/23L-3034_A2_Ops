import argparse
import json
import subprocess
import sys

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score, roc_curve)

from config import OUTPUT_DIR, ROOT, SPLITS_DIR, data_version, git_hash, set_seed, start_mlflow  # noqa: E402
from dataset import get_dataset  # noqa: E402
from model import compile_model, pretrained, simple_cnn, unfreeze_top  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["cnn", "transfer"], default="transfer")
parser.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
parser.add_argument("--img-size", type=int, default=160)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--finetune-lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--finetune-epochs", type=int, default=5)
parser.add_argument("--unfreeze", type=int, default=None, help="0 = skip fine tuning")
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--no-augment", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--autolog", action="store_true")
parser.add_argument("--weights", default="imagenet")
parser.add_argument("--name", default=None)
parser.add_argument("--local", action="store_true")
args = parser.parse_args()


class LogEpochs(keras.callbacks.Callback):
    def __init__(self, mlflow, step_offset=0):
        super().__init__()
        self.mlflow = mlflow
        self.step_offset = step_offset

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        logs["lr"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        self.mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=self.step_offset + epoch)


def pick_threshold(y, p, max_fpr=0.10):
    # best F1 while calling at most 10% of real images "ai"
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        f1 = f1_score(y, pred, zero_division=0)
        if fp / max(fp + tn, 1) <= max_fpr and f1 > best_f1:
            best_t, best_f1 = t, f1
    return round(float(best_t), 2)


def plot_history(hist, best, finetune_start, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, ["loss", "accuracy"]):
        ax.plot(hist[m], label="train")
        ax.plot(hist["val_" + m], label="val")
        ax.axvline(best, color="red", ls="--", label=f"best epoch ({best})")
        if finetune_start:
            ax.axvline(finetune_start - 0.5, color="gray", ls=":", label="fine tuning starts")
        ax.set_title(m)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(run_name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_confusion(cm, thr, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["real", "ai"])
    ax.set_yticks([0, 1], ["real", "ai"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"threshold = {thr}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_roc(y, p, auc, path):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_mistakes(paths, y, p, pred, folder):
    wrong = np.where(pred != y)[0][:10]
    if len(wrong) == 0:
        return
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        with Image.open(ROOT / paths[i]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"#{i} true={'ai' if y[i] else 'real'}  p(ai)={p[i]:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "mistakes.png", dpi=100)
    plt.close(fig)
    pd.DataFrame({"path": paths[wrong], "true": y[wrong], "p_ai": p[wrong]}).to_csv(
        folder / "mistakes.csv", index=False)


set_seed(args.seed)
mlflow = start_mlflow(local=args.local)
run_name = args.name or f"{args.model}_{args.backbone}"
run_dir = OUTPUT_DIR / "runs" / run_name
run_dir.mkdir(parents=True, exist_ok=True)
best_file = str(run_dir / "best.keras")

train_ds = get_dataset("train", args.img_size, args.batch_size, augment=not args.no_augment,
                       shuffle=True, seed=args.seed)
val_ds = get_dataset("val", args.img_size, args.batch_size)
test_ds = get_dataset("test", args.img_size, args.batch_size)

with mlflow.start_run(run_name=run_name) as run:
    mlflow.set_tags({"git_commit": git_hash(), "data_version": data_version(),
                     "seed": args.seed, "logging": "autolog" if args.autolog else "manual"})
    params = {k: v for k, v in vars(args).items() if k not in ("local", "name")}
    # prefix so the names don't clash with the ones autolog adds
    mlflow.log_params({"cfg_" + k: v for k, v in params.items()})
    if args.autolog:
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

    checkpoint = keras.callbacks.ModelCheckpoint(best_file, monitor="val_loss", save_best_only=True)

    def callbacks(step_offset):
        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
            checkpoint,
        ]
        if not args.autolog:
            cbs.append(LogEpochs(mlflow, step_offset))
        return cbs

    finetune_start = None
    if args.model == "cnn":
        model = compile_model(simple_cnn(args.img_size, args.dropout), args.lr)
        hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks(0)).history
    else:
        weights = None if args.weights == "none" else args.weights
        model, base = pretrained(args.backbone, args.img_size, args.dropout, weights)
        trainable = {"stage1": [lyr.name for lyr in model.layers if lyr.trainable and lyr.weights]}
        compile_model(model, args.lr)
        hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks(0)).history

        if args.unfreeze != 0:
            finetune_start = len(hist["loss"])
            trainable["stage2"] = unfreeze_top(base, args.unfreeze)
            compile_model(model, args.finetune_lr)
            if args.autolog:
                # autolog can't log a second fit() in the same run
                with mlflow.start_run(run_name=run_name + "_finetune", nested=True):
                    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                                   callbacks=callbacks(finetune_start)).history
            else:
                h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                               callbacks=callbacks(finetune_start)).history
            hist = {k: hist[k] + h2.get(k, []) for k in hist}
        mlflow.log_dict(trainable, "trainable_layers.json")

    summary = []
    model.summary(print_fn=lambda line, **kw: summary.append(line))
    (run_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))

    layer_rows = []
    for lyr in model.layers:
        try:
            shape = str(tuple(lyr.output.shape))
        except Exception:
            shape = "-"
        layer_rows.append([lyr.name, type(lyr).__name__, shape, lyr.count_params(), lyr.trainable])
    pd.DataFrame(layer_rows, columns=["layer", "type", "output_shape", "params", "trainable"]).to_csv(
        run_dir / "layers.csv", index=False)

    pd.DataFrame({k: v for k, v in hist.items() if k != "learning_rate"}).to_csv(
        run_dir / "history.csv", index_label="epoch")
    best_epoch = int(np.argmin(hist["val_loss"]))
    plot_history(hist, best_epoch, finetune_start, run_dir / "curves.png")

    # evaluate the best checkpoint, test set only used here
    model = keras.models.load_model(best_file)
    y_val = pd.read_csv(SPLITS_DIR / "val.csv").label.values
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    y_test = test_df.label.values
    p_val = model.predict(val_ds, verbose=0).ravel()
    p_test = model.predict(test_ds, verbose=0).ravel()

    thr = pick_threshold(y_val, p_val)
    pred = (p_test >= thr).astype(int)
    results = {
        "best_epoch": best_epoch,
        "best_val_loss": float(min(hist["val_loss"])),
        "val_auc": float(roc_auc_score(y_val, p_val)),
        "threshold": thr,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_auc": float(roc_auc_score(y_test, p_test)),
    }
    mlflow.log_metrics(results)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    plot_confusion(confusion_matrix(y_test, pred, labels=[0, 1]), thr, run_dir / "confusion_matrix.png")
    plot_roc(y_test, p_test, results["test_auc"], run_dir / "roc.png")
    plot_mistakes(test_df.path.values, y_test, p_test, pred, run_dir)

    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (run_dir / "packages.txt").write_text(packages)

    for f in ["curves.png", "confusion_matrix.png", "roc.png", "mistakes.png", "mistakes.csv",
              "summary.txt", "layers.csv", "history.csv", "results.json", "packages.txt"]:
        if (run_dir / f).exists():
            mlflow.log_artifact(str(run_dir / f), "reports")
    mlflow.log_artifacts(str(SPLITS_DIR), "splits")
    mlflow.log_artifact(best_file, "keras_model")
    mlflow.tensorflow.log_model(model, artifact_path="model")
    print("run id:", run.info.run_id)