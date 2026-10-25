import argparse

import keras
import keras_tuner as kt
import pandas as pd

from config import OUTPUT_DIR, set_seed
from dataset import get_dataset
from model import compile_model, pretrained, unfreeze_top

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=8)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--img-size", type=int, default=128)
parser.add_argument("--weights", default="imagenet")
args = parser.parse_args()
set_seed(42)

train_ds = get_dataset("train", args.img_size, augment=True, shuffle=True)
val_ds = get_dataset("val", args.img_size)


def build(hp):
    lr = hp.Choice("learning_rate", [1e-4, 3e-4, 1e-3, 3e-3])
    dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
    n_unfrozen = hp.Choice("unfrozen_layers", [0, 10, 20, 40])
    weights = None if args.weights == "none" else args.weights
    model, base = pretrained("mobilenetv2", args.img_size, dropout, weights)
    if n_unfrozen:
        unfreeze_top(base, n_unfrozen)
    return compile_model(model, lr)


tuner = kt.RandomSearch(build, objective=kt.Objective("val_auc", direction="max"), max_trials=args.trials,
                        seed=42, overwrite=True, directory=str(OUTPUT_DIR / "tuner"), project_name="visionguard")
tuner.search(train_ds, validation_data=val_ds, epochs=args.epochs,
             callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])

trials = [{"trial": t.trial_id, **t.hyperparameters.values, "val_auc": t.score}
          for t in tuner.oracle.trials.values()]
df = pd.DataFrame(trials).sort_values("val_auc", ascending=False)
df.to_csv(OUTPUT_DIR / "tuner" / "trials.csv", index=False)
print(df.to_string(index=False))
print("\nbest:", tuner.get_best_hyperparameters(1)[0].values)

print("\naverage val_auc for each value:")
for col in ["learning_rate", "dropout", "unfrozen_layers"]:
    print(df.groupby(col)["val_auc"].mean().round(4).to_string(), "\n")
