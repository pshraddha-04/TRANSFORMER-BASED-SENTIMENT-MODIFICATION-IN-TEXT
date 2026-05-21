from app.services.emotion_service import EmotionService
import app.services.emotion_service as emotion_module
from app.services.local_emotion_model import LocalEmotionModel


def test_lightweight_uses_token_matching_not_substring():
    service = EmotionService()
    result = service._predict_lightweight("Whatever happened yesterday was odd", top_k=3)

    # "whatever" should not trigger the "hate" keyword by substring overlap.
    assert result.emotion == "neutral"
    assert result.model_source == "lightweight"
    assert "neutral" in result.sentiment_percentages
    assert isinstance(result.sentence_distributions, list)


def test_predict_normalizes_transformer_labels(monkeypatch):
    service = EmotionService()

    monkeypatch.setattr(emotion_module.model_manager, "lightweight_mode", False)

    def _classifier(_text):
        return [[
            {"label": "positive", "score": 0.63},
            {"label": "sadness", "score": 0.2},
            {"label": "neutral", "score": 0.17},
        ]]

    monkeypatch.setattr(emotion_module.model_manager, "get_emotion_components", lambda: (_classifier, object(), object()))
    monkeypatch.setattr(service, "_explain_tokens", lambda text, model, tokenizer: [])

    result = service.predict("I got promoted and I am happy", top_k=3)

    assert result.emotion == "joy"
    assert result.model_source == "transformer"
    assert result.low_confidence is False
    assert result.scores[0].label == "joy"
    assert set(result.sentiment_percentages.keys()) == {"anger", "sadness", "fear", "joy", "surprise", "neutral"}
    assert result.sentence_distributions == [result.sentiment_percentages]


def test_predict_uses_trained_local_model(monkeypatch):
    service = EmotionService()
    local_model = LocalEmotionModel.train(
        [
            ("I am happy and excited", "joy"),
            ("This is awesome", "joy"),
            ("I am sad and upset", "sadness"),
            ("I feel down", "sadness"),
            ("I am worried and anxious", "fear"),
            ("I am afraid", "fear"),
            ("I am angry and frustrated", "anger"),
            ("This is terrible", "anger"),
        ],
        labels=["anger", "sadness", "fear", "joy"],
    )

    monkeypatch.setattr(emotion_module.model_manager, "lightweight_mode", False)
    monkeypatch.setattr(emotion_module.model_manager, "has_local_emotion_model", lambda: True)
    monkeypatch.setattr(emotion_module.model_manager, "get_emotion_components", lambda: (local_model, local_model, None))

    result = service.predict("I am so happy and excited about this", top_k=3)

    assert result.model_source == "local_model"
    assert result.emotion == "joy"
    assert result.confidence > 0.5
    assert result.low_confidence is False


