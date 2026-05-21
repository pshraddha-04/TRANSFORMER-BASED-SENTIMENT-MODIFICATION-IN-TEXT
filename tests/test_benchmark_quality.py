from argparse import Namespace

from scripts.benchmark_quality import (
    _quality_gate_violations,
    compute_classification_metrics,
    load_emotion_dataset,
    load_rewrite_dataset,
)


def test_compute_classification_metrics_basic_shape():
    metrics = compute_classification_metrics(
        y_true=["joy", "anger", "joy", "neutral"],
        y_pred=["joy", "anger", "neutral", "neutral"],
        top_k_hits=3,
        low_conf_hits=1,
    )

    assert metrics["rows"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["top_k_accuracy"] == 0.75
    assert metrics["low_confidence_rate"] == 0.25
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics


def test_load_builtin_benchmark_datasets():
    emotion_rows = load_emotion_dataset("builtin")
    rewrite_rows = load_rewrite_dataset("builtin")

    assert len(emotion_rows) >= 6
    assert len(rewrite_rows) >= 3


def test_load_repo_benchmark_datasets():
    emotion_rows = load_emotion_dataset("data/benchmark_emotion.csv")
    rewrite_rows = load_rewrite_dataset("data/benchmark_rewrite.csv")

    assert len(emotion_rows) == 10
    assert len(rewrite_rows) == 5
    assert {row.label for row in emotion_rows} == {"anger", "sadness", "fear", "joy", "surprise", "neutral"}


def test_load_csv_datasets_from_temp_files(tmp_path):
    emotion_csv = tmp_path / "emotion.csv"
    emotion_csv.write_text("text,label\nI am happy,joy\nI am sad,sadness\n", encoding="utf-8")

    rewrite_csv = tmp_path / "rewrite.csv"
    rewrite_csv.write_text(
        "text,target_tone,user_instruction\nThis is bad,professional,Keep it polite\n",
        encoding="utf-8",
    )

    emotion_rows = load_emotion_dataset(str(emotion_csv))
    rewrite_rows = load_rewrite_dataset(str(rewrite_csv))

    assert len(emotion_rows) == 2
    assert len(rewrite_rows) == 1


def test_quality_gate_violations_detects_failing_metrics():
    report = {
        "emotion": {"accuracy": 0.4, "macro_f1": 0.35, "low_confidence_rate": 0.6},
        "rewrite": {"identity_rate": 0.3, "quality_flag_rate": 0.5, "low_confidence_rate": 0.5},
    }
    args = Namespace(
        min_emotion_accuracy=0.5,
        min_emotion_macro_f1=0.4,
        max_emotion_low_confidence_rate=0.5,
        max_rewrite_identity_rate=0.2,
        max_rewrite_quality_flag_rate=0.4,
        max_rewrite_low_confidence_rate=0.4,
    )

    violations = _quality_gate_violations(report, args)

    assert len(violations) == 6


def test_quality_gate_violations_returns_empty_when_report_is_healthy():
    report = {
        "emotion": {"accuracy": 0.8, "macro_f1": 0.75, "low_confidence_rate": 0.1},
        "rewrite": {"identity_rate": 0.0, "quality_flag_rate": 0.0, "low_confidence_rate": 0.1},
    }
    args = Namespace(
        min_emotion_accuracy=0.5,
        min_emotion_macro_f1=0.4,
        max_emotion_low_confidence_rate=0.5,
        max_rewrite_identity_rate=0.2,
        max_rewrite_quality_flag_rate=0.4,
        max_rewrite_low_confidence_rate=0.4,
    )

    assert _quality_gate_violations(report, args) == []


