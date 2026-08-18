from pydantic import BaseModel
from app.models.paper import Paper

class PaperUploadResponse(BaseModel):
    """Response returned after uploading a paper"""
    message: str
    paper: Paper

class PaperListResponse(BaseModel):
    papers: list[Paper]

class PaperResponse(BaseModel):
    paper: Paper

class PaperDeleteResponse(BaseModel):
    """Response returned after deleting a paper"""
    message : str