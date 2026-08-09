from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_paper_service
from app.models.paper_schemas import PaperUploadResponse
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