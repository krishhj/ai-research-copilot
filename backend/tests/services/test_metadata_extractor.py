import pymupdf

from app.services.metadata_extractor import MetadataExtractor


def test_extract_uses_embedded_metadata(tmp_path):
    pdf_path = tmp_path / "paper.pdf"

    document = pymupdf.open()
    metadata = document.metadata
    metadata["title"] = "Attention Is All You Need"
    metadata["author"] = "Ashish Vaswani; Noam Shazeer"
    document.set_metadata(metadata)

    page = document.new_page()
    page.insert_text(
        (72, 72),
        (
            "Abstract\n"
            "This paper introduces the Transformer architecture.\n"
            "Keywords: transformers, attention, deep learning\n"
            "DOI: 10.1234/example.2025\n"
            "2025"
        ),
    )
    document.save(pdf_path)
    document.close()

    extracted = MetadataExtractor().extract(pdf_path)

    assert extracted.title == "Attention Is All You Need"
    assert extracted.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert extracted.doi == "10.1234/example.2025"
    assert extracted.year == 2025
    assert extracted.keywords == [
        "transformers",
        "attention",
        "deep learning",
    ]


def test_extract_uses_first_page_title_and_authors_as_fallback(tmp_path):
    pdf_path = tmp_path / "paper.pdf"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        (
            "A Practical Guide to RAG\n"
            "Alice Smith and Bob Jones\n"
            "Abstract\n"
            "This paper explains retrieval-augmented generation.\n"
            "Introduction"
        ),
    )
    document.save(pdf_path)
    document.close()

    extracted = MetadataExtractor().extract(pdf_path)

    assert extracted.title == "A Practical Guide to RAG"
    assert extracted.authors == ["Alice Smith", "Bob Jones"]
    assert extracted.abstract == (
        "This paper explains retrieval-augmented generation."
    )