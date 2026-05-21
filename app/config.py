import os
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_list(value: str | None, fallback: list[str]) -> list[str]:
	if not value:
		return fallback
	parsed = [item.strip() for item in value.split(",") if item.strip()]
	return parsed or fallback


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "emotion_model"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map.json"
TRAINING_METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"
MODEL_VERSIONS_DIR = ARTIFACTS_DIR / "model_versions"

APP_NAME = os.getenv("APP_NAME", "Transformer Sentiment Transformation API")
APP_VERSION = os.getenv("APP_VERSION", "1.1.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
IS_PRODUCTION = ENVIRONMENT == "production"

DEFAULT_EMOTION_MODEL = os.getenv("DEFAULT_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
# Use flan-t5-base as the reliable default; the rewrite service will use
# aggressive sampling and prompt tuning to produce visible rewrites.
DEFAULT_REWRITER_MODEL = os.getenv("DEFAULT_REWRITER_MODEL", "google/flan-t5-base")

LIGHTWEIGHT_MODE = _as_bool(os.getenv("LIGHTWEIGHT_MODE"), default=False)
API_KEY = os.getenv("API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

ALLOWED_ORIGINS = _csv_list(
	os.getenv("ALLOWED_ORIGINS"),
	["http://localhost:5173", "http://127.0.0.1:5173"],
)
TRUSTED_HOSTS = _csv_list(
	os.getenv("TRUSTED_HOSTS"),
	["localhost", "127.0.0.1", "testserver"],
)

MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "8"))
MIN_ROWS_PER_LABEL = int(os.getenv("MIN_ROWS_PER_LABEL", "2"))
MIN_UNIQUE_TEXT_RATIO = float(os.getenv("MIN_UNIQUE_TEXT_RATIO", "0.4"))

DRIFT_LOG_EVERY_N = int(os.getenv("DRIFT_LOG_EVERY_N", "25"))
DRIFT_DOMINANCE_THRESHOLD = float(os.getenv("DRIFT_DOMINANCE_THRESHOLD", "0.85"))
DRIFT_WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "100"))

PREDICTION_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_LOW_CONFIDENCE_THRESHOLD", "0.6"))
PREDICTION_FORCE_NEUTRAL_THRESHOLD = float(os.getenv("PREDICTION_FORCE_NEUTRAL_THRESHOLD", "0.4"))
REWRITE_MIN_CONFIDENCE = float(os.getenv("REWRITE_MIN_CONFIDENCE", "0.7"))

