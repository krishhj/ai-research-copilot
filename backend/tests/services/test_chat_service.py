from uuid import uuid4

from app.services.chat_service import ChatService
from app.storage.chroma import VectorSearchResult


def make_result(text: str, page_number: int) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=uuid4(),
        paper_id=uuid4(),
        page_number=page_number,
        text=text,
        score=0.8,
    )


RESULTS = (
    make_result("RAG-Sequence uses the same retrieved document for the whole output.", 3),
    make_result("The retrieval index can be outdated, and swapping it updates knowledge.", 7),
)


def test_adds_marker_to_line_without_one():
    answer = "- RAG-Sequence uses the same retrieved document for the output."

    cited = ChatService._add_missing_citations(answer, RESULTS)

    assert cited == "- RAG-Sequence uses the same retrieved document for the output. [Source 1]"


def test_removes_markers_collected_at_the_end():
    answer = (
        "- RAG-Sequence uses the same retrieved document for the output.\n"
        "- Swapping the outdated retrieval index updates knowledge.\n"
        "\n"
        "[Source 1] [Source 2]"
    )

    cited = ChatService._add_missing_citations(answer, RESULTS)

    assert cited.splitlines() == [
        "- RAG-Sequence uses the same retrieved document for the output. [Source 1]",
        "- Swapping the outdated retrieval index updates knowledge. [Source 2]",
    ]


def test_keeps_existing_markers_and_headings():
    answer = "## Results\n- The index can be outdated [Source 2]."

    assert ChatService._add_missing_citations(answer, RESULTS) == answer


def test_leaves_unrelated_lines_uncited():
    answer = "- These findings show the general value of the approach overall."

    assert ChatService._add_missing_citations(answer, RESULTS) == answer


def test_normalizes_special_spaces_in_markers():
    answer = "- The index can be outdated [Source 2]. It can be swapped [Source 2]."

    assert ChatService._add_missing_citations(answer, RESULTS) == (
        "- The index can be outdated [Source 2]. It can be swapped [Source 2]."
    )


def test_removes_repeated_markers():
    answer = "- RAG-Sequence uses the same retrieved document [Source 1] [Source 1]."

    assert ChatService._add_missing_citations(answer, RESULTS) == (
        "- RAG-Sequence uses the same retrieved document [Source 1]."
    )


def test_does_not_cite_table_header_row():
    answer = (
        "| Retrieved document usage for the whole output | Notes |\n"
        "|---|---|\n"
        "| RAG-Sequence uses the same retrieved document | Whole output |"
    )

    assert ChatService._add_missing_citations(answer, RESULTS).splitlines() == [
        "| Retrieved document usage for the whole output | Notes |",
        "|---|---|",
        "| RAG-Sequence uses the same retrieved document | Whole output [Source 1] |",
    ]
