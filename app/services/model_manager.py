import json
import threading
from typing import Dict

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from app.config import DEFAULT_EMOTION_MODEL, DEFAULT_REWRITER_MODEL, LABEL_MAP_PATH, LIGHTWEIGHT_MODE, MODEL_DIR
from app.services.local_emotion_model import LocalEmotionModel


class ModelManager:
    def __init__(self) -> None:
        self.lightweight_mode = LIGHTWEIGHT_MODE
        self._lock = threading.RLock()
        self._emotion_pipeline = None
        self._emotion_model = None
        self._emotion_tokenizer = None
        self._rewriter_pipeline = None
        self._label_map = None
        self._emotion_model_source = None
        self._rewriter_model_source = None
        self._emotion_model_signature = None

    @staticmethod
    def _inference_device() -> int:
        if torch is None:
            return -1
        return 0 if torch.cuda.is_available() else -1

    def _load_label_map(self) -> Dict[int, str]:
        if self._label_map is not None:
            return self._label_map
        if LABEL_MAP_PATH.exists():
            try:
                raw = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
                self._label_map = {int(k): v for k, v in raw.items()}
            except Exception:
                self._label_map = {}
        else:
            self._label_map = {}
        return self._label_map

    @staticmethod
    def _can_load_local_model(path_candidate) -> bool:
        return bool(
            path_candidate
            and path_candidate.exists()
            and ((path_candidate / "local_emotion_model.json").exists() or (path_candidate / "config.json").exists())
        )

    def _local_emotion_model_path(self):
        return MODEL_DIR / "local_emotion_model.json"

    def _local_emotion_model_signature(self):
        path = self._local_emotion_model_path()
        if not path.exists():
            return None
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)

    def has_local_emotion_model(self) -> bool:
        return self._local_emotion_model_path().exists()

    def model_sources(self) -> Dict[str, str | None]:
        return {
            "emotion_model": self._emotion_model_source,
            "rewriter_model": self._rewriter_model_source,
        }

    def get_emotion_components(self):
        with self._lock:
            local_signature = self._local_emotion_model_signature()
            if local_signature is not None:
                path = self._local_emotion_model_path()
                if (
                    self._emotion_model_source == str(path)
                    and self._emotion_model_signature == local_signature
                    and self._emotion_pipeline is not None
                    and self._emotion_model is not None
                ):
                    return self._emotion_pipeline, self._emotion_model, self._emotion_tokenizer
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    local_model = LocalEmotionModel.from_dict(payload)
                    self._emotion_pipeline = local_model
                    self._emotion_model = local_model
                    self._emotion_tokenizer = None
                    self._emotion_model_source = str(path)
                    self._emotion_model_signature = local_signature
                    return self._emotion_pipeline, self._emotion_model, self._emotion_tokenizer
                except Exception as exc:
                    last_error = exc
            if self.lightweight_mode:
                return None, None, None

            if self._emotion_pipeline is not None and self._emotion_model is not None and self._emotion_tokenizer is not None:
                return self._emotion_pipeline, self._emotion_model, self._emotion_tokenizer

            last_error = None
            label_map = self._load_label_map()
            local_candidate = MODEL_DIR if self._can_load_local_model(MODEL_DIR) else None
            for model_path in (local_candidate, DEFAULT_EMOTION_MODEL):
                if model_path is None:
                    continue
                try:
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                    if label_map:
                        model.config.id2label = label_map
                        model.config.label2id = {v: k for k, v in label_map.items()}

                    text_classifier = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        return_all_scores=True,
                        device=self._inference_device(),
                    )
                    self._emotion_pipeline = text_classifier
                    self._emotion_model = model
                    self._emotion_tokenizer = tokenizer
                    self._emotion_model_source = str(model_path)
                    self._emotion_model_signature = None
                    return self._emotion_pipeline, self._emotion_model, self._emotion_tokenizer
                except Exception as exc:
                    last_error = exc

        raise RuntimeError(f"Unable to load emotion model components: {last_error}")

    def get_rewriter(self):
        if self.lightweight_mode:
            return None

        with self._lock:
            if self._rewriter_pipeline is not None:
                return self._rewriter_pipeline

            try:
                from transformers import pipeline

                self._rewriter_pipeline = pipeline(
                    "text2text-generation",
                    model=DEFAULT_REWRITER_MODEL,
                    tokenizer=DEFAULT_REWRITER_MODEL,
                    device=self._inference_device(),
                )
                self._rewriter_model_source = str(DEFAULT_REWRITER_MODEL)
                return self._rewriter_pipeline
            except Exception as exc:
                raise RuntimeError(f"Unable to load rewrite model: {exc}") from exc


model_manager = ModelManager()

