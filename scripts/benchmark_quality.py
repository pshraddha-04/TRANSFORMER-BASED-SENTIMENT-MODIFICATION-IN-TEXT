"""Quality benchmark for emotion prediction and tone rewriting.

Supports two execution modes:
- service: calls in-process services (no running API server required)
- api: calls HTTP endpoints (/predict and /rewrite)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

ARTIFACTS_DIR = BASE_DIR / "artifacts"
QUALITY_REPORTS_DIR = ARTIFACTS_DIR / "quality_reports"
LATEST_QUALITY_REPORT = ARTIFACTS_DIR / "latest_quality_report.json"
CANONICAL_EMOTIONS = ("anger", "sadness", "fear", "joy", "surprise", "neutral")


@dataclass
class EmotionRow:
    text: str
    label: str


@dataclass
class RewriteRow:
    text: str
    target_tone: str
    user_instruction: str | None


BUILTIN_EMOTION_DATASET = [
    EmotionRow(text="I am furious that my request was ignored.", label="anger"),
    EmotionRow(text="I feel very sad after hearing this news.", label="sadness"),
    EmotionRow(text="I am worried and anxious about tomorrow.", label="fear"),
    EmotionRow(text="I am so happy and excited for this event!", label="joy"),
    EmotionRow(text="Wow, I did not expect this result at all.", label="surprise"),
    EmotionRow(text="I completed the report and sent it to the team.", label="neutral"),
    EmotionRow(text="The delay made me annoyed and frustrated.", label="anger"),
    EmotionRow(text="I feel down and upset today.", label="sadness"),
    EmotionRow(text="I am afraid this plan may fail.", label="fear"),
    EmotionRow(text="This is awesome and I love the outcome.", label="joy"),
]

BUILTIN_REWRITE_DATASET = [
    RewriteRow(
        text="This support is terrible and nobody helped me.",
        target_tone="professional",
        user_instruction="Keep it respectful and concise",
    ),
    RewriteRow(
        text="I hate how this issue was handled.",
        target_tone="empathetic",
        user_instruction="Show understanding and avoid blaming",
    ),
    RewriteRow(
        text="You people are the worst at responding.",
        target_tone="formal",
        user_instruction="Use polite formal phrasing",
    ),
    RewriteRow(
        text="I am scared this deadline will be missed.",
        target_tone="friendly",
        user_instruction="Make it reassuring and constructive",
    ),
    RewriteRow(
        text="The update is okay but communication was unclear.",
        target_tone="concise",
        user_instruction="Keep it under 20 words",
    ),
]


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _fingerprint_rows(rows: list[dict[str, str]]) -> str:
    normalized = "\n".join(
        "\t".join(f"{k}={str(v).strip().casefold()}" for k, v in sorted(row.items()))
        for row in rows
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_csv_rows(csv_path: str) -> list[dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {csv_path}")
        rows = [{k: str(v or "").strip() for k, v in row.items()} for row in reader]

    return [row for row in rows if any(value for value in row.values())]


def load_emotion_dataset(csv_path: str) -> list[EmotionRow]:
    if str(csv_path).strip().casefold() == "builtin":
        return list(BUILTIN_EMOTION_DATASET)

    rows = _load_csv_rows(csv_path)
    required = {"text", "label"}
    if rows and not required.issubset(set(rows[0].keys())):
        raise ValueError(f"Emotion CSV must include columns: {', '.join(sorted(required))}")

    parsed = [EmotionRow(text=row["text"], label=row["label"].casefold()) for row in rows if row["text"] and row["label"]]
    if not parsed:
        raise ValueError("Emotion CSV has no usable rows")
    return parsed


def load_rewrite_dataset(csv_path: str) -> list[RewriteRow]:
    if str(csv_path).strip().casefold() == "builtin":
        return list(BUILTIN_REWRITE_DATASET)

    rows = _load_csv_rows(csv_path)
    required = {"text", "target_tone"}
    if rows and not required.issubset(set(rows[0].keys())):
        raise ValueError(f"Rewrite CSV must include columns: {', '.join(sorted(required))}")

    parsed = [
        RewriteRow(
            text=row["text"],
            target_tone=row["target_tone"],
            user_instruction=row.get("user_instruction") or None,
        )
        for row in rows
        if row["text"] and row["target_tone"]
    ]
    if not parsed:
        raise ValueError("Rewrite CSV has no usable rows")
    return parsed


def compute_classification_metrics(y_true: list[str], y_pred: list[str], top_k_hits: int, low_conf_hits: int) -> dict[str, Any]:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Classification vectors must be non-empty and have equal size")

    labels = sorted(set(y_true) | set(y_pred) | set(CANONICAL_EMOTIONS))
    confusion: dict[str, dict[str, int]] = {label: {pred: 0 for pred in labels} for label in labels}
    for truth, pred in zip(y_true, y_pred):
        confusion[truth][pred] += 1

    per_label: dict[str, dict[str, float]] = {}
    precision_values = []
    recall_values = []
    f1_values = []

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label].values())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_label[label] = {
            "support": float(support),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    total = len(y_true)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    return {
        "rows": total,
        "accuracy": round(correct / total, 4),
        "top_k_accuracy": round(top_k_hits / total, 4),
        "low_confidence_rate": round(low_conf_hits / total, 4),
        "macro_precision": round(sum(precision_values) / len(precision_values), 4),
        "macro_recall": round(sum(recall_values) / len(recall_values), 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "labels": per_label,
        "confusion_matrix": confusion,
    }


def evaluate_emotion_service(rows: list[EmotionRow]) -> dict[str, Any]:
    from app.services.emotion_service import emotion_service

    y_true: list[str] = []
    y_pred: list[str] = []
    source_counter: Counter[str] = Counter()
    top_k_hits = 0
    low_conf_hits = 0

    for row in rows:
        result = emotion_service.predict(row.text, top_k=3)
        predicted = result.emotion.casefold()
        y_true.append(row.label)
        y_pred.append(predicted)
        source_counter[result.model_source] += 1
        top_labels = {score.label.casefold() for score in result.scores[:3]}
        if row.label in top_labels:
            top_k_hits += 1
        if result.low_confidence:
            low_conf_hits += 1

    metrics = compute_classification_metrics(y_true, y_pred, top_k_hits=top_k_hits, low_conf_hits=low_conf_hits)
    metrics["model_source_distribution"] = dict(source_counter)
    return metrics


def evaluate_rewrite_service(rows: list[RewriteRow]) -> dict[str, Any]:
    from app.services.rewrite_service import rewrite_service

    rewrite_source_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()

    identity_count = 0
    length_ratios: list[float] = []
    confidence_scores: list[float] = []

    for row in rows:
        result = rewrite_service.rewrite_with_meta(row.text, row.target_tone, row.user_instruction)
        rewrite_source_counts[result.source] += 1
        if result.fallback_reason:
            fallback_reasons[result.fallback_reason] += 1
        for flag in result.quality_flags:
            quality_flags[flag] += 1

        original_norm = _normalize_text(row.text)
        rewritten_norm = _normalize_text(result.text)
        if original_norm == rewritten_norm:
            identity_count += 1

        original_len = max(1, len(row.text.split()))
        rewritten_len = len(result.text.split())
        length_ratios.append(rewritten_len / original_len)
        confidence_scores.append(float(getattr(result, "confidence", 0.0)))

    total = len(rows)
    avg_ratio = (sum(length_ratios) / len(length_ratios)) if length_ratios else 1.0
    avg_confidence = (sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.0
    low_confidence_rate = (sum(1 for score in confidence_scores if score < 0.7) / total) if total else 0.0

    return {
        "rows": total,
        "identity_rate": round(identity_count / total, 4),
        "avg_length_ratio": round(avg_ratio, 4),
        "rewrite_source_distribution": dict(rewrite_source_counts),
        "fallback_reason_counts": dict(fallback_reasons),
        "quality_flag_counts": dict(quality_flags),
        "quality_flag_rate": round(sum(quality_flags.values()) / total, 4),
        "avg_confidence": round(avg_confidence, 4),
        "low_confidence_rate": round(low_confidence_rate, 4),
    }


def _post_json(url: str, payload: dict[str, Any], api_key: str | None) -> dict[str, Any]:
    import requests

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Request failed {response.status_code}: {response.text[:500]}")
    return response.json()


def evaluate_emotion_api(rows: list[EmotionRow], base_url: str, api_key: str | None) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/predict"

    y_true: list[str] = []
    y_pred: list[str] = []
    source_counter: Counter[str] = Counter()
    top_k_hits = 0
    low_conf_hits = 0

    for row in rows:
        result = _post_json(endpoint, {"text": row.text, "top_k": 3}, api_key=api_key)
        y_true.append(row.label)
        predicted = str(result.get("emotion", "neutral")).casefold()
        y_pred.append(predicted)
        source_counter[str(result.get("model_source", "unknown"))] += 1

        top_labels = {str(score.get("label", "")).casefold() for score in result.get("scores", [])[:3]}
        if row.label in top_labels:
            top_k_hits += 1
        if bool(result.get("low_confidence", False)):
            low_conf_hits += 1

    metrics = compute_classification_metrics(y_true, y_pred, top_k_hits=top_k_hits, low_conf_hits=low_conf_hits)
    metrics["model_source_distribution"] = dict(source_counter)
    return metrics


def evaluate_rewrite_api(rows: list[RewriteRow], base_url: str, api_key: str | None) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/rewrite"

    rewrite_source_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()

    identity_count = 0
    length_ratios: list[float] = []
    confidence_scores: list[float] = []

    for row in rows:
        payload = {
            "text": row.text,
            "target_tone": row.target_tone,
            "user_instruction": row.user_instruction,
        }
        result = _post_json(endpoint, payload, api_key=api_key)

        rewrite_source_counts[str(result.get("rewrite_source", "unknown"))] += 1
        fallback_reason = result.get("fallback_reason")
        if fallback_reason:
            fallback_reasons[str(fallback_reason)] += 1

        for flag in result.get("quality_flags", []):
            quality_flags[str(flag)] += 1

        rewritten_text = str(result.get("rewritten_text", ""))
        if _normalize_text(row.text) == _normalize_text(rewritten_text):
            identity_count += 1

        original_len = max(1, len(row.text.split()))
        rewritten_len = len(rewritten_text.split())
        length_ratios.append(rewritten_len / original_len)
        confidence_scores.append(float(result.get("rewrite_confidence", 0.0)))

    total = len(rows)
    avg_ratio = (sum(length_ratios) / len(length_ratios)) if length_ratios else 1.0
    avg_confidence = (sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.0
    low_confidence_rate = (sum(1 for score in confidence_scores if score < 0.7) / total) if total else 0.0

    return {
        "rows": total,
        "identity_rate": round(identity_count / total, 4),
        "avg_length_ratio": round(avg_ratio, 4),
        "rewrite_source_distribution": dict(rewrite_source_counts),
        "fallback_reason_counts": dict(fallback_reasons),
        "quality_flag_counts": dict(quality_flags),
        "quality_flag_rate": round(sum(quality_flags.values()) / total, 4),
        "avg_confidence": round(avg_confidence, 4),
        "low_confidence_rate": round(low_confidence_rate, 4),
    }


def _quality_gate_violations(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    emotion = report["emotion"]
    rewrite = report["rewrite"]
    violations: list[str] = []

    if emotion["accuracy"] < args.min_emotion_accuracy:
        violations.append(
            f"emotion.accuracy={emotion['accuracy']} < min_emotion_accuracy={args.min_emotion_accuracy}"
        )
    if emotion["macro_f1"] < args.min_emotion_macro_f1:
        violations.append(
            f"emotion.macro_f1={emotion['macro_f1']} < min_emotion_macro_f1={args.min_emotion_macro_f1}"
        )
    if emotion["low_confidence_rate"] > args.max_emotion_low_confidence_rate:
        violations.append(
            "emotion.low_confidence_rate="
            f"{emotion['low_confidence_rate']} > max_emotion_low_confidence_rate={args.max_emotion_low_confidence_rate}"
        )

    if rewrite["identity_rate"] > args.max_rewrite_identity_rate:
        violations.append(
            f"rewrite.identity_rate={rewrite['identity_rate']} > max_rewrite_identity_rate={args.max_rewrite_identity_rate}"
        )
    if rewrite["quality_flag_rate"] > args.max_rewrite_quality_flag_rate:
        violations.append(
            "rewrite.quality_flag_rate="
            f"{rewrite['quality_flag_rate']} > max_rewrite_quality_flag_rate={args.max_rewrite_quality_flag_rate}"
        )
    if rewrite.get("low_confidence_rate", 0.0) > args.max_rewrite_low_confidence_rate:
        violations.append(
            "rewrite.low_confidence_rate="
            f"{rewrite.get('low_confidence_rate', 0.0)} > max_rewrite_low_confidence_rate={args.max_rewrite_low_confidence_rate}"
        )

    return violations


def run_benchmark(
    mode: str,
    emotion_csv: str,
    rewrite_csv: str,
    base_url: str,
    api_key: str | None,
) -> dict[str, Any]:
    emotion_rows = load_emotion_dataset(emotion_csv)
    rewrite_rows = load_rewrite_dataset(rewrite_csv)

    if str(emotion_csv).strip().casefold() == "builtin":
        raw_emotion_rows = [{"text": row.text, "label": row.label} for row in emotion_rows]
    else:
        raw_emotion_rows = _load_csv_rows(emotion_csv)

    if str(rewrite_csv).strip().casefold() == "builtin":
        raw_rewrite_rows = [
            {
                "text": row.text,
                "target_tone": row.target_tone,
                "user_instruction": row.user_instruction or "",
            }
            for row in rewrite_rows
        ]
    else:
        raw_rewrite_rows = _load_csv_rows(rewrite_csv)

    if mode == "service":
        emotion_metrics = evaluate_emotion_service(emotion_rows)
        rewrite_metrics = evaluate_rewrite_service(rewrite_rows)
    elif mode == "api":
        emotion_metrics = evaluate_emotion_api(emotion_rows, base_url=base_url, api_key=api_key)
        rewrite_metrics = evaluate_rewrite_api(rewrite_rows, base_url=base_url, api_key=api_key)
    else:
        raise ValueError("mode must be either 'service' or 'api'")

    return {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "base_url": base_url if mode == "api" else None,
        "datasets": {
            "emotion_csv": "builtin" if str(emotion_csv).strip().casefold() == "builtin" else str(Path(emotion_csv).resolve()),
            "emotion_rows": len(emotion_rows),
            "emotion_fingerprint": _fingerprint_rows(raw_emotion_rows),
            "rewrite_csv": "builtin" if str(rewrite_csv).strip().casefold() == "builtin" else str(Path(rewrite_csv).resolve()),
            "rewrite_rows": len(rewrite_rows),
            "rewrite_fingerprint": _fingerprint_rows(raw_rewrite_rows),
        },
        "emotion": emotion_metrics,
        "rewrite": rewrite_metrics,
    }


def _write_report(report: dict[str, Any]) -> Path:
    QUALITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = QUALITY_REPORTS_DIR / f"quality_{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LATEST_QUALITY_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _print_highlights(report: dict[str, Any], report_path: Path) -> None:
    emotion = report["emotion"]
    rewrite = report["rewrite"]
    print("Quality benchmark completed")
    print(f"- report_path: {report_path}")
    print(f"- mode: {report['mode']}")
    print(f"- emotion.accuracy: {emotion['accuracy']}")
    print(f"- emotion.macro_f1: {emotion['macro_f1']}")
    print(f"- emotion.top_k_accuracy: {emotion['top_k_accuracy']}")
    print(f"- emotion.low_confidence_rate: {emotion['low_confidence_rate']}")
    print(f"- rewrite.identity_rate: {rewrite['identity_rate']}")
    print(f"- rewrite.quality_flag_rate: {rewrite['quality_flag_rate']}")
    print(f"- rewrite.avg_confidence: {rewrite.get('avg_confidence', 0.0)}")
    print(f"- rewrite.low_confidence_rate: {rewrite.get('low_confidence_rate', 0.0)}")
    print(f"- rewrite.fallback_reason_counts: {rewrite['fallback_reason_counts']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark emotion and rewrite quality")
    parser.add_argument("--mode", choices=["service", "api"], default="service", help="Benchmark mode")
    parser.add_argument(
        "--emotion-csv",
        default="builtin",
        help="CSV with columns: text,label or 'builtin'",
    )
    parser.add_argument(
        "--rewrite-csv",
        default="builtin",
        help="CSV with columns: text,target_tone,user_instruction(optional) or 'builtin'",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL when mode=api")
    parser.add_argument("--api-key", default=None, help="Optional X-API-Key header for mode=api")
    parser.add_argument("--min-emotion-accuracy", type=float, default=0.5, help="Fail if emotion accuracy is lower")
    parser.add_argument("--min-emotion-macro-f1", type=float, default=0.4, help="Fail if emotion macro_f1 is lower")
    parser.add_argument(
        "--max-emotion-low-confidence-rate",
        type=float,
        default=0.5,
        help="Fail if emotion low_confidence_rate is higher",
    )
    parser.add_argument("--max-rewrite-identity-rate", type=float, default=0.2, help="Fail if rewrite identity_rate is higher")
    parser.add_argument(
        "--max-rewrite-quality-flag-rate",
        type=float,
        default=0.4,
        help="Fail if rewrite quality_flag_rate is higher",
    )
    parser.add_argument(
        "--max-rewrite-low-confidence-rate",
        type=float,
        default=0.4,
        help="Fail if rewrite low_confidence_rate is higher",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    report = run_benchmark(
        mode=args.mode,
        emotion_csv=args.emotion_csv,
        rewrite_csv=args.rewrite_csv,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    report_path = _write_report(report)
    _print_highlights(report, report_path)
    violations = _quality_gate_violations(report, args)
    if violations:
        print("Quality gates failed:")
        for violation in violations:
            print(f"- {violation}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

