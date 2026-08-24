from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from uuid import UUID

from app.core.exceptions import PDFProcessingError
from app.api.dependencies import get_indexing_service,get_paper_service
from app.models.enums import PaperStatus
from app.models.paper_schemas import PaperListResponse, PaperUploadResponse,PaperDeleteResponse,PaperIndexResponse, PaperResponse
from app.services.indexing_service import IndexingService
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

@router.delete(
    "/{paper_id}",
    response_model=PaperDeleteResponse
)
def delete_paper(paper_id: UUID, paper_service: PaperService = Depends(get_paper_service)) -> PaperDeleteResponse:
    """Delete one uploaded paper"""
    deleted = paper_service.delete_paper(paper_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found."
        )

    return PaperDeleteResponse(message="Paper deleted successfully")


@router.post("/{paper_id}/process", response_model=PaperResponse)
def process_paper(paper_id: UUID, paper_service: PaperService = Depends(get_paper_service)) -> PaperResponse:
    """Extract and store searchable chunks for one uploaded paper"""
    try:
        paper = paper_service.process_paper(paper_id=paper_id)
    except PDFProcessingError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The uploaded file could not be processed as PDF."
        ) from error

    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found."
        )

    return PaperResponse(paper=paper)

@router.post("/{paper_id}/index", response_model=PaperIndexResponse)
def index_paper(
    paper_id: UUID,
    paper_service: PaperService = Depends(get_paper_service),
    indexing_service: IndexingService = Depends(get_indexing_service)
) -> PaperIndexResponse:
    """Create and store vector embeddings for a processed paper"""
    paper = paper_service.get_paper(paper_id=paper_id)

    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found."
        )

    if paper.processing.status != PaperStatus.PROCESSED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Paper must be processed before indexing."
        )

    indexed_chunks = indexing_service.index_paper(paper_id)

    return PaperIndexResponse(
        message="Paper indexed successfully.",
        indexed_chunks=indexed_chunks,
    )