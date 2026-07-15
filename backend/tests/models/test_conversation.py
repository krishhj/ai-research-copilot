from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.models.conversation import Conversation, ConversationMessage
from app.models.enums import ConversationRoles


# Test 1 — Create a Message
def test_create_conversation_message():
    message = ConversationMessage(role=ConversationRoles.USER, content="What is self-attention?")

    assert message.role == ConversationRoles.USER
    assert message.content == "What is self-attention?"


# Test 2 — Message UUID
def test_message_has_uuid():
    message = ConversationMessage(role=ConversationRoles.USER, content="Hello")

    assert isinstance(message.id, UUID)


# Test 3 — Timestamp
def test_message_has_timestamp():
    message = ConversationMessage(role=ConversationRoles.USER, content="Hello")

    assert isinstance(message.timestamp, datetime)


# Test 4 — Empty Content
def test_empty_message_content():
    with pytest.raises(ValidationError):
        ConversationMessage(role=ConversationRoles.USER, content="")


# Test 5 — Create Conversation
def test_create_conversation():
    message = ConversationMessage(role=ConversationRoles.USER, content="Hello")

    conversation = Conversation(messages=[message])

    assert len(conversation.messages) == 1


# Test 6 — Default Messages List
def test_default_messages():
    conversation = Conversation()

    assert conversation.messages == []


# Test 7 — Conversation UUID
def test_conversation_has_uuid():
    conversation = Conversation()

    assert isinstance(conversation.id, UUID)
