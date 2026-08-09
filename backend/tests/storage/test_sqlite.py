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