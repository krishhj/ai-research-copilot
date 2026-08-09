from fastapi import APIRouter

from app.api.routes import health, papers

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(papers.router)