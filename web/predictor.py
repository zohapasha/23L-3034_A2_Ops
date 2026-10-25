import io
import os
import sys
import time

import requests


def load_image(image_bytes, size):
    # copied from src/dataset.py, has to stay the same as training
    import tensorflow as tf
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size))
    img.set_shape((size, size, 3))
    return img


def make_result(p_ai, threshold, version, start):
    is_ai = p_ai >= threshold
    return {
        "label": "AI-generated" if is_ai else "Real",
        "confidence": round(p_ai if is_ai else 1 - p_ai, 4),
        "p_ai": round(p_ai, 4),
        "threshold": threshold,
        "model_version": version,
        "latency_ms": int((time.perf_counter() - start) * 1000),
    }


class LocalModel:
    mode = "local"

    def __init__(self, path, size, threshold, version):
        import keras
        if not os.path.exists(path):
            raise FileNotFoundError(f"model file not found: {path}")
        start = time.perf_counter()
        self.model = keras.models.load_model(path)
        self.size = size
        self.threshold = threshold
        self.version = version
        self.predict(blank_png())  # first prediction is slow, do it now
        print(f"model loaded from {path} in {time.perf_counter() - start:.1f}s", flush=True)

    def predict(self, image_bytes, filename=None):
        start = time.perf_counter()
        x = load_image(image_bytes, self.size)[None]
        p_ai = float(self.model(x, training=False).numpy()[0][0])
        return make_result(p_ai, self.threshold, self.version, start)


class RemoteModel:
    mode = "remote"
    version = "remote"

    def __init__(self, url, api_key):
        if not url:
            raise ValueError("MODEL_API_URL is empty")
        self.url = url.rstrip("/")
        self.api_key = api_key

    def predict(self, image_bytes, filename="upload.jpg"):
        start = time.perf_counter()
        r = requests.post(self.url + "/predict", headers={"X-API-Key": self.api_key},
                          files={"file": (filename, image_bytes)}, timeout=60)
        r.raise_for_status()
        result = r.json()
        result["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return result


def blank_png():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, "PNG")
    return buf.getvalue()


def get_model():
    if os.getenv("INFERENCE_MODE", "local") == "remote":
        return RemoteModel(os.getenv("MODEL_API_URL", ""), os.getenv("MODEL_API_KEY", ""))
    return LocalModel(
        os.getenv("MODEL_PATH", "models/champion.keras"),
        int(os.getenv("IMG_SIZE", 160)),
        float(os.getenv("THRESHOLD", 0.5)),
        os.getenv("MODEL_VERSION", "champion"),
    )


if __name__ == "__main__":
    # quick test: python web/predictor.py image1.jpg image2.jpg
    from dotenv import load_dotenv
    load_dotenv()
    m = get_model()
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            print(path, m.predict(f.read()))