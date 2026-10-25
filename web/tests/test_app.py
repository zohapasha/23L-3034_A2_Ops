import io
import os
import sys

import pytest
from PIL import Image

os.environ["INFERENCE_MODE"] = "remote"
os.environ["MODEL_API_URL"] = "http://localhost:9999"
os.environ["DATABASE_URL"] = "sqlite:///test.db"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app as web  # noqa: E402


class DummyModel:
    mode = "remote"
    version = "test"

    def predict(self, data, name=None):
        return {"label": "AI-generated", "confidence": 0.9, "p_ai": 0.9,
                "threshold": 0.5, "model_version": "test", "latency_ms": 1}


@pytest.fixture
def client():
    web.model = DummyModel()
    web.app.config["TESTING"] = True
    return web.app.test_client()


def make_png():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, "PNG")
    return buf.getvalue()


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["model_loaded"]


def test_good_image(client):
    r = upload(client, make_png(), "cat.png")
    assert r.status_code == 200
    assert b"AI-generated" in r.data
    assert b"cat.png" in client.get("/history").data


def test_wrong_type(client):
    assert upload(client, b"hello", "notes.txt").status_code == 415


def test_fake_image(client):
    assert upload(client, b"not an image", "photo.jpg").status_code == 422


def test_too_big(client):
    big = b"0" * int((web.MAX_MB + 1) * 1024 * 1024)
    assert upload(client, big, "big.jpg").status_code == 413