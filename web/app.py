import io
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from flask import Flask, jsonify, render_template, request  # noqa: E402
from PIL import Image  # noqa: E402
from werkzeug.exceptions import RequestEntityTooLarge  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

import database  # noqa: E402
from predictor import get_model  # noqa: E402

ALLOWED = {"jpg", "jpeg", "png", "bmp"}
MAX_MB = float(os.getenv("MAX_UPLOAD_MB", 5))

app = Flask(__name__)
# flask itself rejects anything really big, the normal limit is checked in check_upload
app.config["MAX_CONTENT_LENGTH"] = int((MAX_MB * 2 + 1) * 1024 * 1024)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev")

model_error = None
try:
    model = get_model()
except Exception as e:
    model = None
    model_error = str(e)
    print("MODEL NOT LOADED:", model_error, flush=True)

database.init_db()


class BadUpload(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.message = message
        self.code = code


def check_upload(file):
    if file is None or file.filename == "":
        raise BadUpload("Please choose an image first.", 400)

    name = secure_filename(file.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED:
        raise BadUpload(f"'.{ext}' files are not allowed. Please upload a JPG, PNG or BMP image.", 415)

    data = file.read()  # kept in memory, never saved to disk
    if not data:
        raise BadUpload("The file is empty.", 400)
    if len(data) > MAX_MB * 1024 * 1024:
        raise BadUpload(f"The file is bigger than {MAX_MB:g} MB.", 413)

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            fmt = img.format
    except Exception:
        raise BadUpload("This file looks like an image but it can't be opened (corrupt or fake).", 422)
    if fmt not in ("JPEG", "PNG", "BMP", "MPO"):
        raise BadUpload(f"{fmt} images are not supported.", 415)
    return data, name


def error_page(message, code):
    return render_template("error.html", message=message, status=code), code


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_MB)


@app.post("/predict")
def predict():
    try:
        data, name = check_upload(request.files.get("image"))
    except BadUpload as e:
        return error_page(e.message, e.code)

    if model is None:
        return error_page("The model is not loaded, check the server logs.", 503)

    try:
        result = model.predict(data, name)
    except Exception as e:
        app.logger.exception("prediction failed")
        return error_page(f"Prediction failed: {e}", 502)

    row_id = database.save(
        filename=name,
        size_kb=round(len(data) / 1024, 1),
        label=result["label"],
        confidence=result["confidence"],
        p_ai=result["p_ai"],
        threshold=result["threshold"],
        model_version=str(result["model_version"]),
        mode=model.mode,
        latency_ms=result["latency_ms"],
    )
    return render_template("result.html", r=result, filename=name, row_id=row_id, mode=model.mode)


@app.get("/history")
def history():
    return render_template("history.html", rows=database.latest(100))


@app.get("/health")
def health():
    db_ok = database.is_up()
    status = "ok" if model and db_ok else "error"
    return jsonify(
        status=status,
        model_loaded=model is not None,
        model_error=model_error,
        mode=os.getenv("INFERENCE_MODE", "local"),
        model_version=getattr(model, "version", None),
        database=db_ok,
        debug=app.debug,
    ), 200 if status == "ok" else 503


@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return error_page(f"The file is bigger than {MAX_MB:g} MB.", 413)


@app.errorhandler(404)
def not_found(e):
    return error_page("Page not found.", 404)


@app.errorhandler(500)
def server_error(e):
    return error_page("Something went wrong on the server.", 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
