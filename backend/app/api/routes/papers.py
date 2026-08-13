from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from uuid import UUID

from app.api.dependencies import get_paper_service
from app.models.paper_schemas import PaperListResponse, PaperUploadResponse, PaperResponse
from app.services.paper_service import PaperService

router = APIRouter(prefix="/papers", tags=["Paper"])

@router.post("", response_model=PaperUploadResponse, status_code= status.HTTP_201_CREATED)
async def upload_paper(
    file: UploadFile = File(...),
    paper_service: PaperService = Depends(get_paper_service)
) -> PaperUploadResponse:
    if file.filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required",
        )

    try:
        paper = paper_service.upload_paper(
            content= await file.read(),
            original_filename= file.filename,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail= str(error)
        ) from error

    return PaperUploadResponse(message= "Paper uploaded successfully", paper=paper)

@router.get(
    "",
    response_model= PaperListResponse,
)
def list_papers(
    paper_service: PaperService = Depends(get_paper_service),
) -> PaperListResponse:
    return PaperListResponse(papers=paper_service.list_papers())

@router.get(
    "/{paper_id}",
    response_model=PaperResponse,
)
def get_paper(
    paper_id: UUID,
    paper_service: PaperService = Depends(get_paper_service)
) -> PaperResponse:
    """Return one uploaded paper by ID"""
    paper = paper_service.get_paper(paper_id)

    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found."
        )

    return PaperResponse(paper=paper)