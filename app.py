import os
import base64
import logging

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

import inference  # noqa: E402  (loads the champion model once, at start-up)
from inference import predict, MODEL_NAME, THRESHOLD  # noqa: E402
from database import (  # noqa: E402
    init_db, log_inference, get_all_inferences, check_db, DB_BACKEND,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("visionguard")

app = Flask(__name__)
# Never hard-code the secret: read it from the environment (random fallback per start).
app.secret_key = os.getenv("SECRET_KEY") or os.urandom(32)

# Task 4.2: Maximum file upload size limit
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MIME_BY_EXT = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "webp": "image/webp", "bmp": "image/bmp"}


def allowed_file(filename):
    return bool(filename) and "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize database (retries while Postgres starts; raises with a clear log if it never does)
init_db()


def _page(**ctx):
    ctx.setdefault("model_name", MODEL_NAME)
    ctx.setdefault("threshold", THRESHOLD)
    return render_template("index.html", **ctx)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return _page()

    # 1. Validation: file presence
    if "file" not in request.files:
        return _page(error="No file part in the upload request."), 400

    file = request.files["file"]
    if not file.filename:
        return _page(error="Please select an image file before clicking Analyze."), 400

    # 2. Validation: extension (reject non-images)
    if not allowed_file(file.filename):
        return _page(error=f"Unsupported file type '{secure_filename(file.filename) or 'unknown'}'. "
                           f"Allowed formats: PNG, JPG, JPEG, WEBP, BMP."), 415

    # Sanitize filename (Sub-task 4.2)
    safe_name = secure_filename(file.filename)
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else "jpeg"

    # 3. Read in memory: the upload is never written to disk (Sub-task 4.2)
    file_bytes = file.read()

    # 4. Decode + predict in memory. A decode failure is the user's problem (422).
    try:
        result = predict(file_bytes)
    except ValueError as e:
        return _page(error=f"Validation Error: {e}"), 422
    except Exception as e:  # noqa: BLE001
        log.exception("Prediction failed")
        return _page(error=f"The model could not process this image ({e})."), 500

    preview = f"data:{MIME_BY_EXT.get(ext, 'image/jpeg')};base64," + \
        base64.b64encode(file_bytes).decode("utf-8")

    # 5. Task 5: permanent database trail. A DB failure must not look like a bad image.
    try:
        log_inference(
            filename=safe_name,
            verdict=result["verdict"],
            confidence=result["confidence"],
            raw_score=result["raw_score"],
            model_name=MODEL_NAME,
            threshold=THRESHOLD,
        )
    except Exception:  # noqa: BLE001
        log.exception("Could not write to the database")

    return _page(prediction=result, preview_img=preview)


@app.route("/history")
def history():
    """Task 5: show past predictions from the database."""
    try:
        records = get_all_inferences(limit=50)
    except Exception:  # noqa: BLE001
        log.exception("Could not read history")
        return render_template("history.html", records=[], model_name=MODEL_NAME,
                               db_backend=DB_BACKEND), 503
    return render_template("history.html", records=records, model_name=MODEL_NAME,
                           db_backend=DB_BACKEND)


@app.route("/health")
def health():
    """Sub-task 4.2 & 4.4: health check used by Docker HEALTHCHECK and Compose.

    Returns 503 when the model is not loaded or the database is unreachable, so a
    broken deployment shows up as 'unhealthy' instead of silently answering 200.
    """
    model_loaded = inference.CHAMPION_MODEL is not None
    db_ok = check_db()
    healthy = model_loaded and db_ok
    return jsonify({
        "status": "healthy" if healthy else "unhealthy",
        "service": "VisionGuard Web App",
        "model_name": MODEL_NAME,
        "model_loaded": model_loaded,
        "threshold": THRESHOLD,
        "database_backend": DB_BACKEND,
        "database_ok": db_ok,
        "max_upload_size_mb": MAX_FILE_SIZE_MB,
    }), 200 if healthy else 503


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic inference."""
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in request body."}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format."}), 400

    safe_name = secure_filename(file.filename)
    file_bytes = file.read()

    try:
        res = predict(file_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:  # noqa: BLE001
        log.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

    try:
        log_inference(
            filename=safe_name,
            verdict=res["verdict"],
            confidence=res["confidence"],
            raw_score=res["raw_score"],
            model_name=MODEL_NAME,
            threshold=THRESHOLD,
        )
    except Exception:  # noqa: BLE001
        log.exception("Could not write to the database")

    return jsonify({"status": "success", "filename": safe_name, **res}), 200


# Friendly error pages (Sub-task 4.2)
@app.errorhandler(413)
def request_entity_too_large(error):
    return _page(error=f"File exceeds the maximum allowed size of {MAX_FILE_SIZE_MB}MB. "
                       f"Please upload a smaller image."), 413


@app.errorhandler(404)
def not_found(error):
    return _page(error="Page not found (404)."), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return _page(error="That action is not allowed on this page (405)."), 405


@app.errorhandler(500)
def server_error(error):
    return _page(error="Something went wrong on our side. Please try again."), 500


if __name__ == "__main__":
    # Task 4.4: Flask server with debug mode OFF unless explicitly enabled by env.
    # Debug mode exposes the Werkzeug interactive debugger (remote code execution
    # for anyone who can reach the port) and restarts on file changes: never in production.
    DEBUG_MODE = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")
    PORT = int(os.getenv("PORT", "5000"))
    print(f"[*] Starting VisionGuard Web App on http://0.0.0.0:{PORT} (debug={DEBUG_MODE})...")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
