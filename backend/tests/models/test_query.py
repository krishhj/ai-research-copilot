from uuid import uuid4
import pytest
from pydantic import ValidationError
from app.models.query import Query

#Test 1 - Valid query creation
def test_create_query():
    """Test creating a valid query."""
    query = Query(
        question="What is self-attention?"
    )

    assert query.question == "What is self-attention?"
    assert query.top_k == 5

#Test 2 - Deafult top k
def test_default_top_k():
    """Test the default value of top_k."""

    query = Query(
        question="Explain transformers."
    )

    assert query.top_k == 5

#Test 3 - Empty questions
def test_empty_question():
    """Test that an empty question is rejected."""

    with pytest.raises(ValidationError):
        Query(
            question=""
        )

#Test 4 - Invalid top k
def test_invalid_top_k():
    """Test that invalid top_k values are rejected."""

    with pytest.raises(ValidationError):
        Query(
            question="Explain attention",
            top_k=0,
        )

#Test 5 - Optional IDs
def test_optional_ids():
    """Test optional paper_id and conversation_id."""

    paper_id = uuid4()
    conversation_id = uuid4()

    query = Query(
        question="Explain positional encoding.",
        paper_id=paper_id,
        conversation_id=conversation_id,
    )

    assert query.paper_id == paper_id
    assert query.conversation_id == conversation_id