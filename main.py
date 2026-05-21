"""Local Uvicorn entrypoint for the FastAPI app."""

from app.api import app
from app.config import HOST, LOG_LEVEL, PORT


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
