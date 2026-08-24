from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.models.answer import Answer
from app.models.query import Query
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model= Answer)
def ask_question(query: Query, chat_service: ChatService = Depends(get_chat_service)) -> Answer:
    """Answer a question using only indexed research-paper content"""
    return chat_service.answer_question(query=query)