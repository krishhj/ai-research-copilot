from uuid import uuid4

from app.services.document_processor import DocumentProcessor
from app.services.pdf_parser import ExtractedPage, ParsedDocument
from app.services.text_cleaner import TextCleaner 
from app.services.text_chunker import TextChunker

class FakePDFParser:
    """Small fake parser used to test the processing workflow"""

    def extract(self, pdf_path):
        return ParsedDocument(
            pages=(
                ExtractedPage(
                    page_number=1,
                    text="  ef\ufb01cient\tattention improves models.  ",
                ),
                ExtractedPage(
                    page_number=2,
                    text="Retrieval finds relevant information.",
                ),
            )
        )

def test_process_returns_clean_page_aware_chunks(tmp_path):
    processor = DocumentProcessor(
        pdf_parser=FakePDFParser(),
        text_cleaner= TextCleaner(),
        text_chunker= TextChunker(max_words=10, overlap_words=0)
    )

    result = processor.process(
        pdf_path=tmp_path/"paper.pdf",
        paper_id=uuid4()
    )

    assert result.total_pages == 2
    assert len(result.chunks) == 2
    assert result.chunks[0].page_number == 1
    assert result.chunks[0].chunk_text == "efficient attention improves models."
    assert result.chunks[1].page_number == 2

def test_process_handles_empty_page(tmp_path):
    processor = DocumentProcessor(
        pdf_parser=FakePDFParser(),
        text_cleaner= TextCleaner(),
        text_chunker= TextChunker(max_words=10, overlap_words=0)
    )

    result = processor.process(
        pdf_path=tmp_path/"paper.pdf",
        paper_id=uuid4()
    )

    assert result.chunks[0].token_count == 4