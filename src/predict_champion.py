import argparse
import os

import mlflow
import tensorflow as tf

from config import MODEL_NAME, start_mlflow
from dataset import load_image

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--local", action="store_true")
args = parser.parse_args()
start_mlflow(local=args.local)

uri = f"models:/{MODEL_NAME}@champion"
print("loading", uri)
model = mlflow.tensorflow.load_model(uri)

size = model.input_shape[1]
x = load_image(tf.io.read_file(args.image), size)[None]
p_ai = float(model.predict(x, verbose=0)[0][0])
thr = float(os.getenv("THRESHOLD", 0.5))

label = "AI-generated" if p_ai >= thr else "Real"
conf = p_ai if p_ai >= thr else 1 - p_ai
print(f"{args.image}\np(ai) = {p_ai:.4f}, threshold = {thr}\nprediction: {label} ({conf:.1%})")