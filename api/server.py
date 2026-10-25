import io
import os
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, jsonify, request
from PIL import Image

MODEL_FILE = os.getenv("MODEL_FILE", "champion.tflite")
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")
API_KEY = os.getenv("API_KEY", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

interpreter = Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()
input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]
IMG_SIZE = int(input_info["shape"][1])
print(f"loaded {MODEL_FILE} (size={IMG_SIZE}, threshold={THRESHOLD})", flush=True)


def resize_bilinear(img, size):
    # same maths as tf.image.resize(method="bilinear") so it matches training
    h, w = img.shape[:2]
    ys = np.clip((np.arange(size) + 0.5) * h / size - 0.5, 0, h - 1)
    xs = np.clip((np.arange(size) + 0.5) * w / size - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bottom = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def load_image(image_bytes, size):
    with Image.open(io.BytesIO(image_bytes)) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return resize_bilinear(arr, size).astype(np.float32)


def predict_p_ai(image_bytes):
    x = load_image(image_bytes, IMG_SIZE)[None]
    interpreter.set_tensor(input_info["index"], x)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_info["index"])[0][0])


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
    p_ai = predict_p_ai(data)
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