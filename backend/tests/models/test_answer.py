from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.answer import Answer, Citation, RetrievedChunk


# Test 1
def test_create_answer():
    citation = Citation(
        paper_id=uuid4(), paper_title="Attention is All You Need", chunk_id=uuid4(), page_number=4
    )

    retrieved_chunk = RetrievedChunk(chunk_id=uuid4(), score=0.92)

    answer = Answer(
        answer_text="Transformer remove recurrence",
        citations=[citation],
        retrieved_chunks=[retrieved_chunk],
        confidence=0.95,
        generation_time=1.25,
    )

    assert answer.answer_text == "Transformer remove recurrence"
    assert answer.confidence == 0.95
    assert len(answer.citations) == 1
    assert len(answer.retrieved_chunks) == 1


# Test 2
def test_empty_answer():
    with pytest.raises(ValidationError):
        Answer(answer_text="")


# Test 3
def test_default_values():
    answer = Answer(answer_text="Hello")

    assert answer.confidence == 0.0
    assert answer.generation_time == 0.0
    assert answer.citations == []
    assert answer.retrieved_chunks == []


# Test 4
def test_invalid_confidence():
    with pytest.raises(ValidationError):
        Answer(
            answer_text="Hello",
            confidence=1.5,
        )


# Test 5
def test_invalid_chunk_score():
    with pytest.raises(ValidationError):
        RetrievedChunk(
            chunk_id=uuid4(),
            score=1.5,
        )


# Test 6
def test_invalid_page_number():
    with pytest.raises(ValidationError):
        Citation(
            paper_id=uuid4(),
            paper_title="Paper",
            chunk_id=uuid4(),
            page_number=0,
        )
