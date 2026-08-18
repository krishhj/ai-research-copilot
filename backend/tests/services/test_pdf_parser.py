import pymupdf
import  pytest

from app.core.exceptions import PDFProcessingError
from app.services.pdf_parser import PDFParser

def create_test_pdf(pdf_path):
    """Create a small PDF for parser tests"""
    document = pymupdf.open()

    first_page = document.new_page()
    first_page.insert_text((72, 72), "First page text")

    second_page = document.new_page()
    second_page.insert_text((72, 72), "Second page text")

    document.save(pdf_path)
    document.close()

def test_extract_returns_text_for_each_page(tmp_path):
    pdf_path = tmp_path/"sample.pdf"
    create_test_pdf(pdf_path)

    parsed_document = PDFParser().extract(pdf_path)

    assert parsed_document.page_count == 2
    assert parsed_document.pages[0].page_number == 1
    assert "First page text" in parsed_document.pages[0].text
    assert parsed_document.pages[1].page_number == 2
    assert "Second page text" in parsed_document.pages[1].text

def test_extract_invalid_pdf_raises_processing_error(tmp_path):
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_bytes(b"This is not a real PDF.")

    with pytest.raises(PDFProcessingError):
        PDFParser().extract(pdf_path)