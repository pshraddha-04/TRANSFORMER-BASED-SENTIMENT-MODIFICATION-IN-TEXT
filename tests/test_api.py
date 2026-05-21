import os

os.environ["LIGHTWEIGHT_MODE"] = "1"

import app.api as api_module
from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert body["docs"] == "/docs"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_lightweight():
    response = client.post("/predict", json={"text": "I am very angry right now", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert "emotion" in body
    assert "scores" in body
    assert "sentiment_percentages" in body
    assert "sentence_distributions" in body
    assert set(body["sentiment_percentages"].keys()) == {"anger", "sadness", "fear", "joy", "surprise", "neutral"}
    assert body["model_source"] == "lightweight"
    assert "low_confidence" in body


def test_rewrite_custom_instruction():
    payload = {
        "text": "This is the worst support ever",
        "target_tone": "professional",
        "user_instruction": "Keep it under 15 words",
    }
    response = client.post("/rewrite", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["applied_tone"] == "professional"
    assert len(body["rewritten_text"]) > 0
    assert body["rewrite_source"] in {"lightweight", "transformer"}
    assert 0.0 <= body["rewrite_confidence"] <= 1.0
    assert "quality_flags" in body


def test_analyze_rewrite():
    payload = {
        "text": "I am scared about the deadline",
        "target_tone": "empathetic",
        "user_instruction": "Make it reassuring",
    }
    response = client.post("/analyze-rewrite", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "rewrite" in body
    assert "rewritten_prediction" in body
    assert body["rewrite"]["applied_tone"] == "empathetic"
    assert "model_source" in body["prediction"]
    assert "sentiment_percentages" in body["prediction"]
    assert "sentence_distributions" in body["prediction"]
    assert "rewrite_source" in body["rewrite"]
    assert "rewrite_confidence" in body["rewrite"]
    assert "emotion" in body["rewritten_prediction"]
    assert isinstance(body["rewritten_prediction"].get("scores", []), list)


def test_tone_suggestions():
    response = client.get("/tones/suggestions", params={"emotion": "anger"})
    assert response.status_code == 200
    body = response.json()
    assert body["emotion"] == "anger"
    assert "professional" in body["suggestions"]


def test_monitoring_snapshot_shape():
    response = client.get("/monitoring/snapshot")
    assert response.status_code == 200
    body = response.json()
    assert "total" in body
    assert "counts" in body
    assert "distribution" in body
    assert "window_total" in body
    assert "window_counts" in body
    assert "window_distribution" in body
    assert "window_size" in body


def test_predict_rejects_blank_text():
    response = client.post("/predict", json={"text": "   ", "top_k": 3})
    assert response.status_code == 422


def test_train_returns_400_for_missing_csv():
    response = client.post(
        "/train",
        json={
            "train_csv_path": "data/does-not-exist.csv",
            "val_csv_path": "data/sample_val.csv",
            "base_model": "distilbert-base-uncased",
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.00002,
            "max_length": 128,
        },
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_api_key_blocks_protected_route(monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", "secret-key")

    response = client.post("/predict", json={"text": "I am angry", "top_k": 3})
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"
    assert response.headers.get("X-Request-Id")
    assert response.headers.get("X-Response-Time-Ms")


def test_api_key_allows_health_without_header(monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", "secret-key")

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_returns_503_on_model_load_failure(monkeypatch):
    monkeypatch.setattr(api_module.model_manager, "lightweight_mode", False)

    def _raise_components_error():
        raise RuntimeError("model load failed")

    monkeypatch.setattr(api_module.model_manager, "get_emotion_components", _raise_components_error)

    response = client.get("/health/ready")
    assert response.status_code == 503
    assert "model load failed" in response.json()["detail"].lower()


def test_monitoring_snapshot_requires_api_key(monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", "secret-key")

    response = client.get("/monitoring/snapshot")
    assert response.status_code == 401


def test_monitoring_snapshot_allows_api_key(monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", "secret-key")

    response = client.get("/monitoring/snapshot", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200


