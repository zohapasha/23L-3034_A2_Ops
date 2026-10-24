import argparse
import sys
from pathlib import Path

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from config import ROOT, start_mlflow  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--local", action="store_true")
args = parser.parse_args()

start_mlflow(local=args.local, experiment="mnist_demo")
keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, y_train = x_train[:5000] / 255.0, y_train[:5000]
x_test, y_test = x_test[:1000] / 255.0, y_test[:1000]

model = keras.Sequential([
    keras.Input((28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(keras.optimizers.Adam(args.lr), "sparse_categorical_crossentropy", metrics=["accuracy"])

with mlflow.start_run(run_name=f"lr_{args.lr}"):
    mlflow.set_tag("type", "demo")
    mlflow.log_params({"learning_rate": args.lr, "epochs": args.epochs, "hidden_units": 64})

    hist = model.fit(x_train, y_train, validation_split=0.2, epochs=args.epochs, verbose=2).history
    for epoch in range(args.epochs):
        mlflow.log_metrics({k: v[epoch] for k, v in hist.items()}, step=epoch)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    mlflow.log_metrics({"test_loss": loss, "test_accuracy": acc})

    plt.plot(hist["loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.title(f"loss (lr={args.lr})")
    plt.legend()
    plot_path = ROOT / "outputs" / f"mnist_loss_{args.lr}.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path)

    mlflow.log_artifact(str(plot_path))
    mlflow.tensorflow.log_model(model, artifact_path="model")
    print("test accuracy:", round(acc, 4))