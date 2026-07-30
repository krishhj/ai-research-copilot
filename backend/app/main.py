from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.exception_handlers import register_exception_handlers
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

tags_metadata = [
    {
        "name": "Health",
        "description": "Health check and system status endpoints.",
    },
    {
        "name": "Papers",
        "description": "Upload and manage research papers.",
    },
    {
        "name": "Chat",
        "description": "Interact with the AI Research Copilot.",
    },
    {
        "name": "Analytics",
        "description": "Retrieve application analytics and statistics.",
    },
]

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description="""
AI Research Copilot Backend API.

This API provides endpoints for:

- Uploading research papers
- Semantic search
- AI-powered chat
- Analytics
""",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    register_exception_handlers(app)
    app.include_router(api_router)
    openapi_tags=tags_metadata,

    return app


app = create_app()


@app.get("/")
def root():
    return {"message": "AI Research Copilot API is running"}
