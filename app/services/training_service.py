import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from app.config import (
    ARTIFACTS_DIR,
    LABEL_MAP_PATH,
    MIN_ROWS_PER_LABEL,
    MIN_TRAIN_ROWS,
    MIN_UNIQUE_TEXT_RATIO,
    MODEL_DIR,
    MODEL_VERSIONS_DIR,
    TRAINING_METADATA_PATH,
)
from app.services.local_emotion_model import LocalEmotionModel


@dataclass
class TrainResult:
    num_labels: int
    labels: list[str]
    metrics: Dict[str, float]


class TrainingService:
    REQUIRED_COLUMNS = {"text", "label"}

    def train(
        self,
        train_csv_path: str,
        val_csv_path: Optional[str],
        base_model: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        max_length: int,
    ) -> TrainResult:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        train_path = Path(train_csv_path)
        if not train_path.exists():
            raise FileNotFoundError(f"Training CSV file not found: {train_csv_path}")

        val_path = Path(val_csv_path) if val_csv_path else None
        if val_path is not None and not val_path.exists():
            raise FileNotFoundError(f"Validation CSV file not found: {val_csv_path}")

        train_rows = self._load_rows(train_csv_path, "Training")
        if not train_rows:
            raise ValueError("Training CSV must contain at least one non-empty row")

        if len(train_rows) < MIN_TRAIN_ROWS:
            raise ValueError(f"Training CSV must include at least {MIN_TRAIN_ROWS} usable rows")

        labels: list[str] = []
        for _, label in train_rows:
            if label not in labels:
                labels.append(label)
        if len(labels) < 2:
            raise ValueError("Training CSV must include at least two distinct labels")

        label_counts = Counter(label for _, label in train_rows)
        scarce_labels = [str(label) for label, count in label_counts.items() if count < MIN_ROWS_PER_LABEL]
        if scarce_labels:
            raise ValueError(
                "Training CSV has underrepresented labels "
                f"(minimum {MIN_ROWS_PER_LABEL} rows per label): {', '.join(scarce_labels)}"
            )

        unique_texts = {text.casefold() for text, _ in train_rows}
        unique_text_ratio = len(unique_texts) / float(len(train_rows))
        if unique_text_ratio < MIN_UNIQUE_TEXT_RATIO:
            raise ValueError(
                "Training CSV has too many duplicate texts "
                f"(unique ratio={unique_text_ratio:.2f}, minimum={MIN_UNIQUE_TEXT_RATIO:.2f})"
            )

        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        if val_csv_path:
            val_rows = self._load_rows(val_csv_path, "Validation")
            val_rows = [row for row in val_rows if row[1] in labels]
            if not val_rows:
                raise ValueError("Validation CSV must contain at least one row whose labels appear in the training data")
        else:
            val_rows = self._stratified_validation_rows(train_rows)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        metrics = {
            "train_rows": float(len(train_rows)),
            "val_rows": float(len(val_rows)),
            "num_labels": float(len(labels)),
            "unique_text_ratio": float(round(unique_text_ratio, 4)),
            "largest_label_share": float(round(max(label_counts.values()) / len(train_rows), 4)),
        }

        local_model = LocalEmotionModel.train(train_rows, labels)
        local_model_path = MODEL_DIR / "local_emotion_model.json"
        local_model_path.write_text(json.dumps(local_model.to_dict(), indent=2), encoding="utf-8")

        with LABEL_MAP_PATH.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({str(k): v for k, v in id2label.items()}, indent=2))

        run_id = uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()
        metadata: dict[str, object] = {
            "run_id": run_id,
            "created_at_utc": created_at,
            "base_model": base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "num_labels": len(labels),
            "labels": labels,
            "train_csv_path": str(Path(train_csv_path).resolve()),
            "val_csv_path": str(Path(val_csv_path).resolve()) if val_csv_path else None,
            "train_dataset_fingerprint": self._dataset_fingerprint(train_rows),
            "val_dataset_fingerprint": self._dataset_fingerprint(val_rows),
            "label_distribution": {label: int(label_counts[label]) for label in labels},
            "model_artifact_path": str(local_model_path.resolve()),
            "quality_gates": {
                "min_train_rows": MIN_TRAIN_ROWS,
                "min_rows_per_label": MIN_ROWS_PER_LABEL,
                "min_unique_text_ratio": MIN_UNIQUE_TEXT_RATIO,
            },
            "metrics": metrics,
        }

        with TRAINING_METADATA_PATH.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, indent=2))

        MODEL_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        version_file = MODEL_VERSIONS_DIR / f"{run_id}.json"
        version_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return TrainResult(
            num_labels=len(labels),
            labels=labels,
            metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (float, int))},
        )

    def _load_rows(self, csv_path: str, kind: str) -> list[tuple[str, str]]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"{kind} CSV file not found: {csv_path}")
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or not self.REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
                    raise ValueError(f"{kind} CSV must include columns: text,label")
                rows = []
                for row in reader:
                    rows.append((str(row.get("text", "")).strip(), str(row.get("label", "")).strip()))
        except Exception as exc:
            raise ValueError(f"Unable to read {kind.lower()} CSV: {exc}") from exc

        return [(text, label) for text, label in rows if text and label]

    def _dataset_fingerprint(self, rows: list[tuple[str, str]]) -> str:
        normalized = "\n".join(f"{text.casefold()}\t{label.casefold()}" for text, label in rows)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _stratified_validation_rows(self, rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for text, label in rows:
            grouped[label].append((text, label))

        validation_rows: list[tuple[str, str]] = []
        for label_rows in grouped.values():
            take_count = max(1, len(label_rows) // 5)
            validation_rows.extend(label_rows[:take_count])

        # Keep validation set bounded while preserving at least one row per label.
        max_val_rows = min(40, max(len(grouped), len(rows) // 5))
        if len(validation_rows) > max_val_rows:
            validation_rows = validation_rows[:max_val_rows]
        return validation_rows


training_service = TrainingService()
