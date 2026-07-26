from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.constants import APP_NAME, APP_VERSION
from app.core.logger import logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""

    logging.info("Starting AI Research Copilot")
    # Future Initializations here
    yield

    logging.info("Shutting down AI Research Copilot")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description="AI Research Copilot Backend API",
        lifespan=lifespan,
    )

    app.include_router(api_router)

    return app


app = create_app()


@app.get("/")
def root():
    return {"message": "AI Research Copilot API is running"}
