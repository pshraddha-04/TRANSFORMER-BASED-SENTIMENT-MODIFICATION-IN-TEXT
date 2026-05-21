from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class EmotionScore(BaseModel):
    label: str
    score: float


class ExplainToken(BaseModel):
    token: str
    importance: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    emotion: str
    confidence: float
    scores: List[EmotionScore]
    sentiment_percentages: Dict[str, float]
    sentence_distributions: List[Dict[str, float]] = Field(default_factory=list)
    explanation_tokens: List[ExplainToken]
    model_source: str = "lightweight"
    low_confidence: bool = False


class RewriteRequest(BaseModel):
    text: str = Field(min_length=1)
    target_tone: str = Field(default="professional")
    user_instruction: Optional[str] = None

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value

    @field_validator("target_tone")
    @classmethod
    def normalize_tone(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("target_tone must not be blank")
        return value

    @field_validator("user_instruction")
    @classmethod
    def normalize_instruction(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class RewriteResponse(BaseModel):
    original_text: str
    detected_emotion: str
    rewritten_text: str
    applied_tone: str
    suggestions: List[str]
    rewrite_source: str = "lightweight"
    rewrite_confidence: float = Field(ge=0.0, le=1.0)
    quality_flags: List[str] = Field(default_factory=list)
    fallback_reason: Optional[str] = None


class TrainRequest(BaseModel):
    train_csv_path: str
    val_csv_path: Optional[str] = None
    base_model: str = "distilbert-base-uncased"
    epochs: int = Field(default=1, ge=1, le=10)
    batch_size: int = Field(default=8, ge=1, le=64)
    learning_rate: float = Field(default=2e-5, gt=0)
    max_length: int = Field(default=128, ge=32, le=512)

    @field_validator("train_csv_path", "val_csv_path", "base_model")
    @classmethod
    def normalize_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class TrainResponse(BaseModel):
    status: str
    num_labels: int
    labels: List[str]
    metrics: Dict[str, float]


class CombinedRequest(BaseModel):
    text: str = Field(min_length=1)
    target_tone: str = Field(default="professional")
    user_instruction: Optional[str] = None

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value

    @field_validator("target_tone")
    @classmethod
    def normalize_tone(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("target_tone must not be blank")
        return value

    @field_validator("user_instruction")
    @classmethod
    def normalize_instruction(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class CombinedResponse(BaseModel):
    prediction: PredictResponse
    rewrite: RewriteResponse
    rewritten_prediction: Optional[PredictResponse] = None


class MonitoringSnapshotResponse(BaseModel):
    total: int
    counts: Dict[str, int]
    distribution: Dict[str, float]
    window_total: int
    window_counts: Dict[str, int]
    window_distribution: Dict[str, float]
    window_size: int


