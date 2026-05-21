import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.schemas import (
    CombinedRequest,
    CombinedResponse,
    MonitoringSnapshotResponse,
    PredictRequest,
    PredictResponse,
    RewriteRequest,
    RewriteResponse,
    TrainRequest,
    TrainResponse,
)
from app.config import (
    ALLOWED_ORIGINS,
    API_KEY,
    APP_NAME,
    APP_VERSION,
    ENVIRONMENT,
    IS_PRODUCTION,
    LOG_LEVEL,
    TRUSTED_HOSTS,
)
from app.services.emotion_service import emotion_service
from app.services.model_manager import model_manager
from app.services.monitoring_service import prediction_monitor
from app.services.rewrite_service import rewrite_service
from app.services.training_service import training_service

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app.api")


app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-Id"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)

AUTH_EXEMPT_PATHS = {"/", "/health", "/health/ready", "/docs", "/redoc", "/openapi.json"}


def _is_exempt_path(path: str) -> bool:
    if path in AUTH_EXEMPT_PATHS:
        return True
    return path.startswith("/docs/") or path.startswith("/redoc/")


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    started = time.perf_counter()

    if API_KEY and request.method != "OPTIONS":
        if not _is_exempt_path(request.url.path):
            supplied_key = request.headers.get("X-API-Key")
            if supplied_key != API_KEY:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                logger.warning("[%s] Unauthorized request to %s", request_id, request.url.path)
                response = JSONResponse(status_code=401, content={"detail": "Unauthorized"})
                response.headers["X-Request-Id"] = request_id
                response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
                return response

    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    logger.info("[%s] %s %s -> %s (%.2f ms)", request_id, request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def _safe_error_detail(exc: Exception) -> str:
    if IS_PRODUCTION:
        return "Internal server error"
    return str(exc)


@app.get("/")
def root() -> dict:
    return {
        "message": APP_NAME,
        "environment": ENVIRONMENT,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "ready": "/health/ready",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": ENVIRONMENT, "version": APP_VERSION}


@app.get("/health/ready")
def readiness() -> dict:
    if model_manager.lightweight_mode:
        return {
            "status": "ready",
            "lightweight_mode": True,
            "model_sources": model_manager.model_sources(),
        }

    try:
        model_manager.get_emotion_components()
        model_manager.get_rewriter()
        return {
            "status": "ready",
            "lightweight_mode": False,
            "model_sources": model_manager.model_sources(),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=_safe_error_detail(exc)) from exc


@app.get("/monitoring/snapshot", response_model=MonitoringSnapshotResponse)
def monitoring_snapshot() -> MonitoringSnapshotResponse:
    try:
        return MonitoringSnapshotResponse(**prediction_monitor.snapshot())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error_detail(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result = emotion_service.predict(req.text, req.top_k)
        prediction_monitor.record_prediction(result.emotion)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error_detail(exc)) from exc


@app.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest) -> RewriteResponse:
    try:
        prediction = emotion_service.predict(req.text, top_k=3)
        prediction_monitor.record_prediction(prediction.emotion)
        rewritten = rewrite_service.rewrite_with_meta(req.text, req.target_tone, req.user_instruction)
        suggestions = rewrite_service.tone_suggestions(prediction.emotion)
        return RewriteResponse(
            original_text=req.text,
            detected_emotion=prediction.emotion,
            rewritten_text=rewritten.text,
            applied_tone=req.target_tone,
            suggestions=suggestions,
            rewrite_source=rewritten.source,
            rewrite_confidence=rewritten.confidence,
            quality_flags=rewritten.quality_flags,
            fallback_reason=rewritten.fallback_reason,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error_detail(exc)) from exc


@app.post("/analyze-rewrite", response_model=CombinedResponse)
def analyze_and_rewrite(req: CombinedRequest) -> CombinedResponse:
    try:
        prediction = emotion_service.predict(req.text, top_k=3)
        prediction_monitor.record_prediction(prediction.emotion)
        rewritten = rewrite_service.rewrite_with_meta(req.text, req.target_tone, req.user_instruction)
        rewritten_prediction = emotion_service.predict(rewritten.text, top_k=3)
        prediction_monitor.record_prediction(rewritten_prediction.emotion)
        suggestions = rewrite_service.tone_suggestions(prediction.emotion)

        return CombinedResponse(
            prediction=prediction,
            rewrite=RewriteResponse(
                original_text=req.text,
                detected_emotion=prediction.emotion,
                rewritten_text=rewritten.text,
                applied_tone=req.target_tone,
                suggestions=suggestions,
                rewrite_source=rewritten.source,
                rewrite_confidence=rewritten.confidence,
                quality_flags=rewritten.quality_flags,
                fallback_reason=rewritten.fallback_reason,
            ),
            rewritten_prediction=rewritten_prediction,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error_detail(exc)) from exc


@app.get("/tones/suggestions")
def tone_suggestions(emotion: str) -> dict:
    try:
        return {"emotion": emotion, "suggestions": rewrite_service.tone_suggestions(emotion)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error_detail(exc)) from exc


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    try:
        result = training_service.train(
            train_csv_path=req.train_csv_path,
            val_csv_path=req.val_csv_path,
            base_model=req.base_model,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            max_length=req.max_length,
        )
        return TrainResponse(status="completed", num_labels=result.num_labels, labels=result.labels, metrics=result.metrics)
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=_safe_error_detail(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_safe_error_detail(exc)) from exc

