import app.services.rewrite_service as rewrite_module
from app.services.rewrite_service import RewriteService


def test_rewrite_falls_back_when_transformer_repeats(monkeypatch):
    service = RewriteService()
    monkeypatch.setattr(rewrite_module.model_manager, "lightweight_mode", False)

    class RepeatingRewriter:
        def __call__(self, prompt, **kwargs):
            return [{"generated_text": "This support is bad. This support is bad."}]

    monkeypatch.setattr(rewrite_module.model_manager, "get_rewriter", lambda: RepeatingRewriter())

    result = service.rewrite_with_meta("This support is terrible", "professional", None)

    assert result.source == "lightweight"
    assert result.fallback_reason == "transformer_low_quality_output"
    assert 0.0 <= result.confidence <= 1.0
    assert result.quality_flags == []
    assert "poor" in result.text.lower() or "frustrating" in result.text.lower()
    assert result.text.endswith(".")


def test_rewrite_uses_transformer_output_when_quality_is_ok(monkeypatch):
    service = RewriteService()
    monkeypatch.setattr(rewrite_module.model_manager, "lightweight_mode", False)

    class GoodRewriter:
        def __call__(self, prompt, **kwargs):
            return [{"generated_text": "This support is poor."}]

    monkeypatch.setattr(rewrite_module.model_manager, "get_rewriter", lambda: GoodRewriter())

    result = service.rewrite_with_meta("This support is terrible", "professional", None)

    assert result.source == "transformer"
    assert result.fallback_reason is None
    assert result.confidence > 0.7
    assert result.quality_flags == []
    assert result.text.lower() == "this support is poor."


def test_rewrite_repairs_identity_output_without_fallback(monkeypatch):
    service = RewriteService()
    monkeypatch.setattr(rewrite_module.model_manager, "lightweight_mode", False)

    class IdentityRewriter:
        def __call__(self, prompt, **kwargs):
            return [{"generated_text": "This support is terrible"}]

    monkeypatch.setattr(rewrite_module.model_manager, "get_rewriter", lambda: IdentityRewriter())

    result = service.rewrite_with_meta("This support is terrible", "professional", "Keep it concise")

    assert result.source == "transformer"
    assert result.fallback_reason is None
    assert result.confidence >= 0.78
    assert result.quality_flags == []
    assert result.text.endswith(".")
    assert "terrible" not in result.text.lower()


def test_rewrite_falls_back_when_transformer_confidence_is_too_low(monkeypatch):
    service = RewriteService()
    monkeypatch.setattr(rewrite_module.model_manager, "lightweight_mode", False)

    class BareMinimumRewriter:
        def __call__(self, prompt, **kwargs):
            return [{"generated_text": "This support is poor."}]

    monkeypatch.setattr(rewrite_module.model_manager, "get_rewriter", lambda: BareMinimumRewriter())
    monkeypatch.setattr(service, "_rewrite_confidence", lambda original, rewritten, flags: 0.45)

    result = service.rewrite_with_meta("This support is terrible", "professional", None)

    assert result.source == "lightweight"
    assert result.fallback_reason == "transformer_low_confidence"


def test_rewrite_keeps_transformer_output_for_soft_quality_flags(monkeypatch):
    service = RewriteService()
    monkeypatch.setattr(rewrite_module.model_manager, "lightweight_mode", False)

    class SoftFlagRewriter:
        def __call__(self, prompt, **kwargs):
            return [{"generated_text": "The update is acceptable."}]

    monkeypatch.setattr(rewrite_module.model_manager, "get_rewriter", lambda: SoftFlagRewriter())
    monkeypatch.setattr(service, "_quality_flags", lambda original, rewritten: ["length_drift"])
    monkeypatch.setattr(service, "_rewrite_confidence", lambda original, rewritten, flags: 0.82)

    result = service.rewrite_with_meta("The update is okay.", "formal", None)

    assert result.source == "transformer"
    assert result.fallback_reason is None
    assert result.quality_flags == ["length_drift"]
    assert result.confidence == 0.82


def test_rewrite_falls_back_when_transformer_has_critical_quality_flags(monkeypatch):
    service = RewriteService()
    monkeypatch.setattr(rewrite_module.model_manager, "lightweight_mode", False)

    class CriticalFlagRewriter:
        def __call__(self, prompt, **kwargs):
            return [{"generated_text": "I can help with this."}]

    monkeypatch.setattr(rewrite_module.model_manager, "get_rewriter", lambda: CriticalFlagRewriter())
    monkeypatch.setattr(service, "_quality_flags", lambda original, rewritten: ["negation_mismatch"])

    result = service.rewrite_with_meta("I do not want this changed.", "professional", None)

    assert result.source == "lightweight"
    assert result.fallback_reason == "transformer_low_quality_output"


def test_rewrite_falls_back_on_negation_change(monkeypatch):
    service = RewriteService()
    monkeypatch.setattr(rewrite_module.model_manager, "lightweight_mode", False)

    class NegationFlippingRewriter:
        def __call__(self, prompt, **kwargs):
            return [{"generated_text": "I can help with this quickly."}]

    monkeypatch.setattr(rewrite_module.model_manager, "get_rewriter", lambda: NegationFlippingRewriter())

    result = service.rewrite_with_meta("I do not want this changed.", "professional", None)

    assert result.source == "lightweight"
    assert result.fallback_reason == "transformer_low_quality_output"


def test_lightweight_rewrite_changes_common_complaint_patterns():
    service = RewriteService()

    result = service._rewrite_lightweight("I hate how this issue was handled.", "empathetic", None)

    assert result != "I hate how this issue was handled."
    assert "hate" not in result.lower()
    assert result.endswith(".")


def test_lightweight_rewrite_reframes_near_identity_input():
    service = RewriteService()

    result = service._rewrite_lightweight("This support is terrible", "professional", None)

    assert result != "This support is terrible"
    assert "terrible" not in result.lower()
    assert "not satisfactory" in result.lower() or "poor" in result.lower()


