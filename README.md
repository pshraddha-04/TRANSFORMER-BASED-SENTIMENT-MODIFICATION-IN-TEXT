# A Transformer-Based Framework for Controlled Sentiment Transformation in Text 

REST API for AI-powered emotion detection, explainable predictions, and controlled tone rewriting.

## Features
- Transformer-based emotion classification (`/predict`)
- Token importance explanation for each prediction
- Controlled rewriting with custom target tone and optional user instruction (`/rewrite`)
- Combined detect + rewrite endpoint (`/analyze-rewrite`)
- Tone suggestions based on detected emotion (`/tones/suggestions`)

## Project Structure
```
Sem6_2.0/
├── app/
│   ├── api.py                  # FastAPI endpoints
│   ├── config.py               # Environment config
│   ├── schemas.py              # Request/response models
│   └── services/
│       ├── emotion_service.py  # Prediction + explanation
│       └── rewrite_service.py  # Controlled rewriting
├── tests/
│   └── test_api.py             # Smoke tests
├── main.py                     # Local API runner
├── requirements.txt
└── .env.example                # Environment variable template
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create and activate virtual environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure environment
```powershell
copy .env.example .env
```
Edit `.env` with your values before running in production.

### 5. Run the API
```powershell
python main.py
```

Open docs at `http://127.0.0.1:8000/docs`.

## Lightweight Mode (no model downloads)
```powershell
$env:LIGHTWEIGHT_MODE="1"
python main.py
```

Uses keyword-based heuristics — `/predict`, `/rewrite`, `/analyze-rewrite`, and `/tones/suggestions` all work without downloading transformer models.

## API Examples

### Predict emotion
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body '{"text":"I am very upset about this delay.","top_k":3}'
```

### Check service health
```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

### Rewrite tone
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/rewrite" -ContentType "application/json" -Body '{"text":"This support is terrible","target_tone":"professional","user_instruction":"keep it to one sentence"}'
```

### Analyze then rewrite
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/analyze-rewrite" -ContentType "application/json" -Body '{"text":"I am scared about the deadline","target_tone":"empathetic","user_instruction":"make it reassuring"}'
```

## Production Environment Variables
```powershell
$env:ENVIRONMENT="production"
$env:TF_USE_LEGACY_KERAS="1"
$env:ALLOWED_ORIGINS="https://your-frontend.example.com"
$env:TRUSTED_HOSTS="api.example.com"
$env:API_KEY="replace-with-a-long-random-secret"
$env:LOG_LEVEL="INFO"
python main.py
```

- `API_KEY` enables header auth (`X-API-Key`) for non-health endpoints.
- `ALLOWED_ORIGINS` is comma-separated and controls CORS.
- `TRUSTED_HOSTS` is comma-separated and protects Host header handling.
- `TF_USE_LEGACY_KERAS=1` helps transformer rewrite loading on environments with Keras 3.
- `/health` is liveness; `/health/ready` validates model readiness.
- `GET /monitoring/snapshot` returns prediction monitoring counters (protected when `API_KEY` is enabled).

## Model Controls
- Prediction traffic is tracked for drift hints and logs periodic label-distribution snapshots.
- Drift checks use a rolling window (configure with `DRIFT_WINDOW_SIZE`, default `100`).
- Confidence gates are configurable via `PREDICTION_LOW_CONFIDENCE_THRESHOLD`, `PREDICTION_FORCE_NEUTRAL_THRESHOLD`, and `REWRITE_MIN_CONFIDENCE`.

## Response Details
- Prediction responses include `sentiment_percentages` for all emotions and optional `sentence_distributions` for sentence-by-sentence visibility.
- Rewrite responses include `rewrite_confidence` to judge output trustworthiness.
- Rewrite preserves facts, numbers, negation, and subject as much as possible.

## Notes
- GPU is recommended for inference speed.
- If transformer checkpoints cannot be loaded, services fall back to lightweight behavior automatically.
- Never commit `.env` — use `.env.example` as the template.
