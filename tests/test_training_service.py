import csv
import json

import pytest

import app.services.training_service as training_module
from app.services.training_service import TrainingService


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


def test_train_writes_versioned_metadata(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    model_dir = artifacts_dir / "emotion_model"
    label_map_path = artifacts_dir / "label_map.json"
    versions_dir = artifacts_dir / "model_versions"
    latest_metadata_path = artifacts_dir / "training_metadata.json"

    monkeypatch.setattr(training_module, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(training_module, "MODEL_DIR", model_dir)
    monkeypatch.setattr(training_module, "LABEL_MAP_PATH", label_map_path)
    monkeypatch.setattr(training_module, "MODEL_VERSIONS_DIR", versions_dir)
    monkeypatch.setattr(training_module, "TRAINING_METADATA_PATH", latest_metadata_path)

    train_csv = tmp_path / "train.csv"
    _write_csv(
        train_csv,
        [
            {"text": "I am happy", "label": "joy"},
            {"text": "This is awesome", "label": "joy"},
            {"text": "I feel sad", "label": "sadness"},
            {"text": "This is upsetting", "label": "sadness"},
            {"text": "I am worried", "label": "fear"},
            {"text": "I am anxious", "label": "fear"},
            {"text": "I am angry", "label": "anger"},
            {"text": "This is frustrating", "label": "anger"},
        ],
    )

    service = TrainingService()
    result = service.train(
        train_csv_path=str(train_csv),
        val_csv_path=None,
        base_model="distilbert-base-uncased",
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        max_length=128,
    )

    assert result.num_labels == 4
    assert latest_metadata_path.exists()
    assert label_map_path.exists()
    assert (model_dir / "local_emotion_model.json").exists()

    version_files = list(versions_dir.glob("*.json"))
    assert len(version_files) == 1

    metadata = json.loads(latest_metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"]
    assert metadata["train_dataset_fingerprint"]
    assert metadata["metrics"]["unique_text_ratio"] >= 0.4
    assert metadata["model_artifact_path"].endswith("local_emotion_model.json")


def test_train_accepts_bundled_starter_csvs(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(training_module, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(training_module, "MODEL_DIR", artifacts_dir / "emotion_model")
    monkeypatch.setattr(training_module, "LABEL_MAP_PATH", artifacts_dir / "label_map.json")
    monkeypatch.setattr(training_module, "MODEL_VERSIONS_DIR", artifacts_dir / "model_versions")
    monkeypatch.setattr(training_module, "TRAINING_METADATA_PATH", artifacts_dir / "training_metadata.json")

    service = TrainingService()
    result = service.train(
        train_csv_path="data/sample_train.csv",
        val_csv_path="data/sample_val.csv",
        base_model="distilbert-base-uncased",
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        max_length=128,
    )

    assert result.num_labels == 6
    assert set(result.labels) == {"anger", "sadness", "fear", "joy", "surprise", "neutral"}
    assert (artifacts_dir / "training_metadata.json").exists()
    assert (artifacts_dir / "label_map.json").exists()
    assert list((artifacts_dir / "model_versions").glob("*.json"))


def test_train_rejects_low_unique_text_ratio(tmp_path):
    train_csv = tmp_path / "train_low_unique.csv"
    _write_csv(
        train_csv,
        [
            {"text": "Same text", "label": "joy"},
            {"text": "Same text", "label": "joy"},
            {"text": "Same text", "label": "sadness"},
            {"text": "Same text", "label": "sadness"},
            {"text": "Same text", "label": "fear"},
            {"text": "Same text", "label": "fear"},
            {"text": "Different", "label": "anger"},
            {"text": "Different", "label": "anger"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        TrainingService().train(
            train_csv_path=str(train_csv),
            val_csv_path=None,
            base_model="distilbert-base-uncased",
            epochs=1,
            batch_size=8,
            learning_rate=2e-5,
            max_length=128,
        )

