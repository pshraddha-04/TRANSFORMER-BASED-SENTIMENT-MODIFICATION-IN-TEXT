from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Sequence

TOKEN_RE = re.compile(r"\b[a-z']+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def extract_features(text: str) -> list[str]:
    tokens = tokenize(text)
    features = list(tokens)
    if len(tokens) >= 2:
        features.extend(f"{tokens[i]}__{tokens[i + 1]}" for i in range(len(tokens) - 1))
    if len(tokens) >= 3:
        features.extend(f"{tokens[i]}__{tokens[i + 1]}__{tokens[i + 2]}" for i in range(len(tokens) - 2))
    return features


@dataclass
class LocalEmotionModel:
    labels: list[str]
    label_priors: Dict[str, float]
    feature_counts: Dict[str, Dict[str, int]]
    feature_totals: Dict[str, int]
    vocabulary_size: int
    alpha: float = 1.0

    @classmethod
    def train(cls, rows: Sequence[tuple[str, str]], labels: Sequence[str], alpha: float = 1.0) -> "LocalEmotionModel":
        label_order = list(labels)
        label_counts = Counter(label for _, label in rows)
        feature_counts: Dict[str, Counter[str]] = {label: Counter() for label in label_order}
        feature_totals: Dict[str, int] = {label: 0 for label in label_order}
        vocabulary: set[str] = set()

        for text, label in rows:
            if label not in feature_counts:
                continue
            features = extract_features(text)
            if not features:
                continue
            counts = Counter(features)
            feature_counts[label].update(counts)
            feature_totals[label] += sum(counts.values())
            vocabulary.update(counts.keys())

        total_rows = sum(label_counts[label] for label in label_order) or 1
        label_priors = {
            label: (label_counts[label] + alpha) / (total_rows + alpha * len(label_order))
            for label in label_order
        }

        return cls(
            labels=label_order,
            label_priors=label_priors,
            feature_counts={label: dict(counts) for label, counts in feature_counts.items()},
            feature_totals=feature_totals,
            vocabulary_size=max(1, len(vocabulary)),
            alpha=alpha,
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "LocalEmotionModel":
        raw_labels = payload.get("labels")
        labels = [str(label) for label in raw_labels] if isinstance(raw_labels, (list, tuple)) else []

        label_priors: Dict[str, float] = {}
        raw_priors = payload.get("label_priors")
        if isinstance(raw_priors, dict):
            label_priors = {str(k): float(v) for k, v in raw_priors.items()}

        feature_counts: Dict[str, Dict[str, int]] = {}
        raw_feature_counts = payload.get("feature_counts")
        if isinstance(raw_feature_counts, dict):
            feature_counts = {
                str(label): {str(token): int(count) for token, count in counts.items()}
                for label, counts in raw_feature_counts.items()
                if isinstance(counts, dict)
            }

        feature_totals: Dict[str, int] = {}
        raw_feature_totals = payload.get("feature_totals")
        if isinstance(raw_feature_totals, dict):
            feature_totals = {str(label): int(total) for label, total in raw_feature_totals.items()}

        raw_vocab_size = payload.get("vocabulary_size")
        vocabulary_size = int(raw_vocab_size) if isinstance(raw_vocab_size, (int, float, str)) else 1
        raw_alpha = payload.get("alpha")
        alpha = float(raw_alpha) if isinstance(raw_alpha, (int, float, str)) else 1.0
        return cls(labels, label_priors, feature_counts, feature_totals, vocabulary_size, alpha)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": "local_multinomial_naive_bayes",
            "labels": self.labels,
            "label_priors": self.label_priors,
            "feature_counts": self.feature_counts,
            "feature_totals": self.feature_totals,
            "vocabulary_size": self.vocabulary_size,
            "alpha": self.alpha,
        }

    def predict_distribution(self, text: str) -> Dict[str, float]:
        if not self.labels:
            return {}

        features = Counter(extract_features(text))
        log_scores: Dict[str, float] = {}
        for label in self.labels:
            prior = self.label_priors.get(label, 1.0 / len(self.labels))
            total = self.feature_totals.get(label, 0)
            label_counts = self.feature_counts.get(label, {})
            denom = total + self.alpha * self.vocabulary_size
            score = math.log(max(prior, 1e-12))
            for feature, count in features.items():
                feature_count = label_counts.get(feature, 0)
                prob = (feature_count + self.alpha) / max(denom, 1e-12)
                score += count * math.log(max(prob, 1e-12))
            log_scores[label] = score

        max_log = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_log) for label, score in log_scores.items()}
        total_score = sum(exp_scores.values()) or 1.0
        return {label: value / total_score for label, value in exp_scores.items()}

    def __call__(self, text: str):
        distribution = self.predict_distribution(text)
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        return [[{"label": label, "score": score} for label, score in ranked]]

