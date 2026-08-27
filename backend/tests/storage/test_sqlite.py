from uuid import uuid4

from app.models.chunk import Chunk
from app.models.enums import PaperStatus
from app.models.paper import Paper, PaperMetaData, ProcessingMetadata
from app.storage.sqlite import SQLitePaperRepository

def test_add_and_list_papers(tmp_path):
    repository = SQLitePaperRepository(
        database_path= tmp_path / "papers.db"
    )

    paper = Paper(
        metadata=PaperMetaData(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            year = 2017
        ),
        processing=ProcessingMetadata(
            stored_filename="attention.pdf"
        )
    )

    repository.add(paper=paper)

    papers = repository.list_all()

    assert len(papers) == 1
    assert papers[0].id == paper.id
    assert papers[0].metadata.title == "Attention Is All You Need"
    assert papers[0].metadata.authors == ["Ashish Vaswani"]
    assert papers[0].processing.stored_filename == "attention.pdf"

def test_save_processing_result_saves_chunks_and_updates_paper(tmp_path):
    repository = SQLitePaperRepository(
        database_path=tmp_path / "papers.db",
    )

    paper = Paper(
        metadata=PaperMetaData(title="Attention Is All You Need"),
        processing=ProcessingMetadata(
            stored_filename="attention.pdf",
        ),
    )
    repository.add(paper)

    chunks = (
        Chunk(
            paper_id=paper.id,
            chunk_index=0,
            page_number=1,
            chunk_text="Attention is all you need.",
            token_count=5,
        ),
        Chunk(
            paper_id=paper.id,
            chunk_index=1,
            page_number=2,
            chunk_text="Transformers use self-attention.",
            token_count=4,
        ),
    )

    repository.save_processing_result(
        paper_id=paper.id,
        total_pages=2,
        chunks=chunks,
        metadata=paper.metadata,
    )

    saved_paper = repository.get_by_id(paper.id)
    saved_chunks = repository.list_chunks_by_paper_id(paper.id)

    assert saved_paper is not None
    assert saved_paper.processing.status == PaperStatus.PROCESSED
    assert saved_paper.processing.total_pages == 2
    assert saved_paper.processing.total_chunks == 2
    assert len(saved_chunks) == 2
    assert saved_chunks[0].chunk_text == "Attention is all you need."