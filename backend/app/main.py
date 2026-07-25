from fastapi import FastAPI

from app.core.constants import APP_NAME, APP_VERSION


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description="AI Research Copilot Backend API",
    )

    return app


app = create_app()
