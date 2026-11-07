"""Automated tests for VisionGuard (Task 6.6 CI). They need no dataset and no DagsHub access."""
import io

import numpy as np
import pytest
from PIL import Image

import inference
from app import app
from database import get_all_inferences, init_db, log_inference

MODEL_LOADED = inference.CHAMPION_MODEL is not None
needs_model = pytest.mark.skipif(not MODEL_LOADED, reason="champion model file not available")


def make_image_bytes(fmt="JPEG", mode="RGB", size=(200, 150)):
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype("uint8")
    img = Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return app.test_client()


# ---------- inference module (Task 4.1) ----------
def test_preprocess_shape_and_dtype():
    batch = inference.preprocess_image_bytes(make_image_bytes())
    assert batch.shape == (1, 128, 128, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 255.0  # raw pixels; scaling is inside the model


@pytest.mark.parametrize("fmt,mode", [("PNG", "RGBA"), ("PNG", "L"), ("JPEG", "RGB")])
def test_preprocess_handles_modes(fmt, mode):
    batch = inference.preprocess_image_bytes(make_image_bytes(fmt, mode))
    assert batch.shape == (1, 128, 128, 3)


def test_preprocess_rejects_garbage():
    with pytest.raises(ValueError):
        inference.preprocess_image_bytes(b"this is not an image")


# ---------- routes and validation (Task 4.2) ----------
def test_health_reports_status(client):
    res = client.get("/health")
    body = res.get_json()
    assert {"status", "model_loaded", "database_ok", "database_backend"} <= set(body)
    assert body["database_ok"] is True
    assert res.status_code == (200 if MODEL_LOADED else 503)


def test_index_and_history_pages_load(client):
    assert client.get("/").status_code == 200
    assert client.get("/history").status_code == 200


def test_rejects_wrong_file_type(client):
    res = client.post("/", data={"file": (io.BytesIO(b"hello"), "notes.txt")},
                      content_type="multipart/form-data")
    assert res.status_code == 415
    assert b"Unsupported file type" in res.data


def test_rejects_fake_image(client):
    res = client.post("/", data={"file": (io.BytesIO(b"not really a jpeg"), "fake.jpg")},
                      content_type="multipart/form-data")
    assert res.status_code in (422, 500)  # 500 only if the model is missing
    if MODEL_LOADED:
        assert res.status_code == 422
        assert b"decode" in res.data.lower()


def test_rejects_oversized_upload(client):
    big = io.BytesIO(b"X" * (6 * 1024 * 1024))
    res = client.post("/", data={"file": (big, "huge.jpg")}, content_type="multipart/form-data")
    assert res.status_code == 413
    assert b"maximum allowed size" in res.data


def test_unknown_page_is_friendly(client):
    res = client.get("/does-not-exist")
    assert res.status_code == 404
    assert b"404" in res.data


# ---------- database trail (Task 5) ----------
def test_database_roundtrip():
    init_db()
    log_inference("unit.png", "Real Photography", 91.5, 0.085, "unit_model", 0.5)
    rows = get_all_inferences(limit=1)
    assert rows[0]["filename"] == "unit.png"
    assert rows[0]["verdict"] == "Real Photography"
    assert rows[0]["model_name"] == "unit_model"


@needs_model
def test_valid_upload_is_predicted_logged_and_name_sanitised(client):
    before = len(get_all_inferences(limit=1000))
    res = client.post("/", data={"file": (io.BytesIO(make_image_bytes()), "../../evil name.jpg")},
                      content_type="multipart/form-data")
    assert res.status_code == 200
    assert b"Confidence" in res.data or b"confidence" in res.data
    rows = get_all_inferences(limit=1000)
    assert len(rows) == before + 1
    assert "/" not in rows[0]["filename"] and ".." not in rows[0]["filename"]


@needs_model
def test_api_predict_returns_json(client):
    res = client.post("/api/predict", data={"file": (io.BytesIO(make_image_bytes()), "x.jpg")},
                      content_type="multipart/form-data")
    body = res.get_json()
    assert res.status_code == 200
    assert body["verdict"] in ("AI-Generated Image", "Real Photography")
    assert 0.0 <= body["confidence"] <= 100.0
