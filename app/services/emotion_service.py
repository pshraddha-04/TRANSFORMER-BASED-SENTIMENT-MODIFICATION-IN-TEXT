import re
from typing import Dict, List

# torch is optional — only needed for attention-based explanation
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from app.schemas import EmotionScore, ExplainToken, PredictResponse
from app.config import PREDICTION_FORCE_NEUTRAL_THRESHOLD, PREDICTION_LOW_CONFIDENCE_THRESHOLD
from app.services.model_manager import model_manager


class EmotionService:
    # The 6 emotion labels this service always outputs
    CANONICAL_LABELS = ("anger", "sadness", "fear", "joy", "surprise", "neutral")

    # Maps raw model output labels to canonical labels
    LABEL_ALIASES = {
        "anger": "anger",
        "angry": "anger",
        "sad": "sadness",
        "sadness": "sadness",
        "fear": "fear",
        "fearful": "fear",
        "anxiety": "fear",
        "joy": "joy",
        "happiness": "joy",
        "happy": "joy",
        "positive": "joy",
        "surprise": "surprise",
        "surprised": "surprise",
        "neutral": "neutral",
    }

    # Single-word emotion keywords — each match adds +0.45 to that emotion's score
    KEYWORD_RULES = {
        "anger": ["angry", "furious", "hate", "annoyed", "stupid", "frustrated", "irritated", "ignored"],
        "sadness": ["sad", "down", "depressed", "upset", "cry", "disappointed", "letdown", "miserable"],
        "fear": ["scared", "afraid", "anxious", "panic", "worried", "nervous", "concerned"],
        "joy": ["happy", "great", "awesome", "excited", "love", "pleased", "delighted"],
        "surprise": ["surprised", "wow", "unexpected", "shocked"],
        "neutral": [],
    }

    # Multi-word phrases — stronger signal than single keywords (+0.85 per match)
    PHRASE_RULES = {
        "anger": ["fed up", "not acceptable", "wasted my time", "nobody helped", "worst at"],
        "sadness": ["feel down", "felt down", "let down", "very sad", "so disappointed"],
        "fear": ["worried about", "afraid this", "scared this", "may fail", "might fail"],
        "joy": ["so happy", "really happy", "excited for", "love the outcome", "great news"],
        "surprise": ["did not expect", "didn't expect", "wow", "at all"],
    }

    # Flat set of all keywords — used for fast lookup during explanation
    KEYWORDS = {word for words in KEYWORD_RULES.values() for word in words}

    def predict(self, text: str, top_k: int = 3) -> PredictResponse:
        """Main entry point. Tries transformer first, falls back to rule-based if unavailable."""
        top_k = max(1, int(top_k or 1))
        # Split text into sentences for per-sentence scoring
        segments = self._sentence_segments(text)

        # Skip transformer entirely if in lightweight mode with no local model
        if model_manager.lightweight_mode and not model_manager.has_local_emotion_model():
            return self._predict_lightweight(text, top_k, segments)

        try:
            classifier, model, tokenizer = model_manager.get_emotion_components()
            if classifier is None:
                return self._predict_lightweight(text, top_k, segments)

            # Score each sentence using transformer + rule blend, then aggregate
            ranked, sentence_distributions = self._aggregate_sentence_predictions(
                segments,
                lambda segment: self._predict_classifier_distribution(segment, classifier),
            )
            if not ranked:
                return self._predict_lightweight(text, top_k, segments)

            # Use attention-based explanation if full model is available, else fall back to keywords
            if model is not None and tokenizer is not None:
                explanation = self._explain_tokens(text=text, model=model, tokenizer=tokenizer)
                model_source = "transformer"
            else:
                explanation = self._explain_keywords(text)
                model_source = "local_model" if model_manager.has_local_emotion_model() else "lightweight"
            top = ranked[:top_k]
            best_label, best_score = self._apply_low_confidence_guard(top[0][0], float(top[0][1]))
            return PredictResponse(
                emotion=best_label,
                confidence=best_score,
                scores=[EmotionScore(label=label, score=float(round(score, 4))) for label, score in top],
                sentiment_percentages=self._to_percentages(dict(ranked)),
                sentence_distributions=sentence_distributions,
                explanation_tokens=explanation,
                model_source=model_source,
                low_confidence=best_score < PREDICTION_LOW_CONFIDENCE_THRESHOLD,
            )
        except Exception:
            return self._predict_lightweight(text, top_k, segments)

    def _predict_lightweight(self, text: str, top_k: int, segments: object = None) -> PredictResponse:
        """Fallback path — scores text using only keyword/phrase rules, no model needed."""
        if isinstance(segments, list):
            segment_list = segments
        else:
            segment_list = self._sentence_segments(text)
        ranked, sentence_distributions = self._aggregate_sentence_predictions(segment_list, self._predict_lightweight_distribution)
        # If scoring produced nothing, distribute equally across all labels
        if not ranked:
            ranked = [(label, 1.0 / len(self.CANONICAL_LABELS)) for label in self.CANONICAL_LABELS]

        explanation = self._explain_keywords(text)
        best_label, best_score = self._apply_low_confidence_guard(ranked[0][0], float(ranked[0][1]))

        return PredictResponse(
            emotion=best_label,
            confidence=float(round(min(best_score, 0.99), 4)),
            scores=[EmotionScore(label=l, score=float(round(s, 4))) for l, s in ranked[:top_k]],
            sentiment_percentages=self._to_percentages(dict(ranked)),
            sentence_distributions=sentence_distributions,
            explanation_tokens=explanation,
            model_source="lightweight",
            low_confidence=best_score < PREDICTION_LOW_CONFIDENCE_THRESHOLD,
        )

    def _sentence_segments(self, text: str) -> List[str]:
        """Splits text into sentences on '.', '!', '?' boundaries."""
        segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]
        return segments or [text.strip()]

    def _normalize_distribution(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Converts raw scores to probabilities that sum to 1 across all canonical labels."""
        total = sum(scores.values())
        if total <= 0:
            return {label: 1.0 / len(self.CANONICAL_LABELS) for label in self.CANONICAL_LABELS}
        return {label: scores.get(label, 0.0) / total for label in self.CANONICAL_LABELS}

    def _aggregate_sentence_predictions(self, segments: List[str] | None, scorer) -> tuple[List[tuple[str, float]], List[Dict[str, float]]]:
        """Scores each sentence, weights by sqrt(word count), returns averaged final distribution."""
        if not segments:
            return [], []

        aggregate = {label: 0.0 for label in self.CANONICAL_LABELS}
        sentence_distributions: List[Dict[str, float]] = []
        total_weight = 0.0

        for segment in segments:
            segment_scores = scorer(segment)
            normalized = self._normalize_distribution(segment_scores)
            sentence_distributions.append(self._to_percentages(normalized))
            # Longer sentences get more influence — weight = sqrt(word count)
            weight = max(1.0, float(len(segment.split()) ** 0.5))
            total_weight += weight
            for label in self.CANONICAL_LABELS:
                aggregate[label] += normalized.get(label, 0.0) * weight

        if total_weight <= 0:
            return [], sentence_distributions

        ranked = sorted(((label, score / total_weight) for label, score in aggregate.items()), key=lambda x: x[1], reverse=True)
        return ranked, sentence_distributions

    def _predict_transformer_distribution(self, segment: str, classifier) -> Dict[str, float]:
        """Runs the HuggingFace classifier and maps its output to canonical labels."""
        outputs = classifier(segment)[0]
        normalized_scores = {label: 0.0 for label in self.CANONICAL_LABELS}
        for output in outputs or []:
            label = self._normalize_label(str(output.get("label", "")))
            normalized_scores[label] += float(output.get("score", 0.0))
        return normalized_scores

    def _predict_classifier_distribution(self, segment: str, classifier) -> Dict[str, float]:
        """Blends transformer scores (78%) with rule-based scores (22%) to reduce overconfidence."""
        classifier_scores = self._predict_transformer_distribution(segment, classifier)
        heuristic_scores = self._predict_lightweight_distribution(segment)
        return self._blend_distributions(classifier_scores, heuristic_scores, primary_weight=0.78)

    def _predict_lightweight_distribution(self, segment: str) -> Dict[str, float]:
        """Pure rule-based scorer — no model. Uses keyword matches, phrase matches, and negation handling."""
        tokens = re.findall(r"\b[a-z']+\b", segment.lower())
        token_set = set(tokens)
        scores = {label: 0.0 for label in self.CANONICAL_LABELS}
        normalized_text = " ".join(tokens)

        # Single keyword matches — moderate signal
        for label, words in self.KEYWORD_RULES.items():
            for word in words:
                if word in token_set:
                    scores[label] += 0.45

        # Phrase matches — stronger signal than single keywords
        for label, phrases in self.PHRASE_RULES.items():
            for phrase in phrases:
                if phrase in normalized_text:
                    scores[label] += 0.85 if " " in phrase else 0.55

        # Negation handling — flip or dampen scores when negation words are present
        if any(marker in token_set for marker in {"not", "never", "no", "n't"}):
            if any(word in token_set for word in {"happy", "great", "love", "excited", "awesome"}):
                scores["joy"] *= 0.4
                scores["sadness"] += 0.15
            if any(word in token_set for word in {"expect", "expected"}) and any(word in token_set for word in {"did", "didn't", "didnt"}):
                scores["surprise"] += 0.35
            if any(word in token_set for word in {"help", "helped", "helping"}) and any(word in token_set for word in {"nobody", "noone", "no"}):
                scores["anger"] += 0.35

        emotional_scores = {label: scores[label] for label in self.CANONICAL_LABELS if label != "neutral"}
        # Exclamation mark boosts the already-dominant emotion
        if "!" in segment and emotional_scores and max(emotional_scores.values()) > 0:
            dominant = max((label for label in self.CANONICAL_LABELS if label != "neutral"), key=lambda label: emotional_scores[label])
            scores[dominant] += 0.18

        # No signals found → strongly neutral; otherwise neutral score shrinks as emotion strength grows
        if all(v == 0.0 for v in scores.values()):
            scores["neutral"] = 0.92
        else:
            strongest = max(scores[label] for label in self.CANONICAL_LABELS if label != "neutral")
            scores["neutral"] = max(0.02, 0.32 - strongest * 0.12)
        return scores

    def _to_percentages(self, score_map: Dict[str, float]) -> Dict[str, float]:
        # Keep a stable shape so clients always receive all canonical labels.
        percentages: Dict[str, float] = {}
        for label in self.CANONICAL_LABELS:
            percentages[label] = float(round(score_map.get(label, 0.0) * 100.0, 2))
        return percentages

    def _normalize_label(self, raw_label: str) -> str:
        """Cleans and maps a raw model label string to one of the 6 canonical labels."""
        label = raw_label.strip().lower().replace("-", "_")
        label = label.replace("label_", "")
        if label.isdigit():
            return "neutral"
        label = label.replace("_", " ").strip()
        return self.LABEL_ALIASES.get(label, "neutral")

    def _apply_low_confidence_guard(self, label: str, confidence: float) -> tuple[str, float]:
        """Forces prediction to neutral if confidence is too low to trust any specific emotion."""
        if label != "neutral" and confidence < (PREDICTION_FORCE_NEUTRAL_THRESHOLD * 0.6):
            return "neutral", 0.5
        return label, confidence

    def _blend_distributions(
        self,
        primary: Dict[str, float],
        secondary: Dict[str, float],
        primary_weight: float = 0.75,
    ) -> Dict[str, float]:
        """Weighted average of two score distributions, both normalized before blending."""
        primary_normalized = self._normalize_distribution(primary)
        secondary_normalized = self._normalize_distribution(secondary)
        blended = {
            label: (primary_normalized.get(label, 0.0) * primary_weight)
            + (secondary_normalized.get(label, 0.0) * (1.0 - primary_weight))
            for label in self.CANONICAL_LABELS
        }
        return self._normalize_distribution(blended)

    def _explain_keywords(self, text: str) -> List[ExplainToken]:
        """Rule-based explanation — returns up to 5 matched emotion keywords, all with importance=1.0."""
        tokens = re.findall(r"\b[a-z']+\b", text.lower())
        matched = []
        seen = set()
        for token in tokens:
            if token in self.KEYWORDS and token not in seen:
                matched.append(token)
                seen.add(token)
        return [ExplainToken(token=t, importance=1.0) for t in matched[:5]]

    def _explain_tokens(self, text: str, model, tokenizer) -> List[ExplainToken]:
        """Transformer explanation — uses CLS token attention from the last layer to rank token importance."""
        if torch is None:
            return []

        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model(**encoded, output_attentions=True)

        # Last layer attention, averaged across all heads, CLS token row
        attentions = out.attentions[-1]
        cls_attention = attentions[0, :, 0, :].mean(dim=0)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

        pairs = []
        for token, weight in zip(tokens, cls_attention.tolist()):
            if token in {"[CLS]", "[SEP]", "<s>", "</s>", "<pad>"}:
                continue
            clean = token.replace("##", "")
            pairs.append((clean, float(weight)))

        pairs.sort(key=lambda x: x[1], reverse=True)
        return [ExplainToken(token=t, importance=round(w, 4)) for t, w in pairs[:6]]


emotion_service = EmotionService()
