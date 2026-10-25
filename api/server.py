import io
import os
import time

import keras
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image

MODEL_PATH = os.getenv("MODEL_PATH", "champion.keras")
IMG_SIZE = int(os.getenv("IMG_SIZE", 160))
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")
API_KEY = os.getenv("API_KEY", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

model = keras.models.load_model(MODEL_PATH)
print(f"loaded {MODEL_PATH} (size={IMG_SIZE}, threshold={THRESHOLD})", flush=True)


def load_image(image_bytes, size):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size))
    img.set_shape((size, size, 3))
    return img


@app.get("/")
@app.get("/health")
def health():
    return jsonify(status="ok", model_version=MODEL_VERSION, img_size=IMG_SIZE, threshold=THRESHOLD)


@app.post("/predict")
def predict():
    if not API_KEY or request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="missing or wrong X-API-Key"), 401

    file = request.files.get("file")
    if file is None:
        return jsonify(error="send the image in a form field called 'file'"), 400
    data = file.read()
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception:
        return jsonify(error="not a valid image"), 422

    start = time.perf_counter()
    p_ai = float(model(load_image(data, IMG_SIZE)[None], training=False).numpy()[0][0])
    is_ai = p_ai >= THRESHOLD
    return jsonify(
        label="AI-generated" if is_ai else "Real",
        confidence=round(p_ai if is_ai else 1 - p_ai, 4),
        p_ai=round(p_ai, 4),
        threshold=THRESHOLD,
        model_version=MODEL_VERSION,
        inference_ms=int((time.perf_counter() - start) * 1000),
    )


@app.errorhandler(413)
def too_large(e):
    return jsonify(error="file too large"), 413