from pydantic import BaseModel
from app.models.paper import Paper

class PaperUploadResponse(BaseModel):
    """Response returned after uplaoding a paper"""
    message: str
    paper: Paper

class PaperListResponse(BaseModel):
    papers: list[Paper]

class PaperResponse(BaseModel):
    paper: Paper