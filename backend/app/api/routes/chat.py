from fastapi import APIRouter, Depends

from app.core.auth import AuthenticatedUser, get_current_user
from app.api.dependencies import get_chat_service
from app.models.answer import Answer
from app.models.query import Query
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model= Answer)
def ask_question(query: Query, chat_service: ChatService = Depends(get_chat_service), current_user: AuthenticatedUser = Depends(get_current_user) ) -> Answer:
    """Answer a question using only indexed research-paper content"""
    return chat_service.answer_question(query=query, owner_id=current_user.id)