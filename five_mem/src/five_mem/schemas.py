from datetime import datetime
from typing import List, Optional, Literal
from uuid import UUID

import pydantic


class EventEntity(pydantic.BaseModel):
    """Base entity with timestamp fields"""
    created_at: datetime = pydantic.Field(
        description="Timestamp when the entity was created"
    )
    updated_at: datetime = pydantic.Field(
        description="Timestamp when the entity was last updated"
    )
    deleted_at: Optional[datetime] = pydantic.Field(
        default=None,
        description="Timestamp when the entity was deleted (soft delete)"
    )


class Message(pydantic.BaseModel):
    """Individual message in a conversation"""
    id: UUID = pydantic.Field(
        description="Unique identifier for the message"
    )
    user_id: UUID = pydantic.Field(
        description="Identifier of the user who owns this session"
    )
    session_id: UUID = pydantic.Field(
        description="Identifier of the conversation session"
    )
    author: Literal["user", "assistant"] = pydantic.Field(
        description="Author of the message (user or assistant)"
    )
    turn: int = pydantic.Field(
        description="Turn number in the conversation sequence"
    )
    payload: str = pydantic.Field(
        description="Content of the message"
    )
