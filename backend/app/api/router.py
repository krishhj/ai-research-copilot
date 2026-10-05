from fastapi import APIRouter

from app.api.routes import auth, chat, health, papers, search

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(health.router)
api_router.include_router(papers.router)
api_router.include_router(search.router)
api_router.include_router(chat.router)