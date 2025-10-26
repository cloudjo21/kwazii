from typing import Optional, Literal

import pydantic

from five_mem import schemas


class SessionMessageHistory(pydantic.BaseModel):
    """A list of messages from a session."""
    messages: list[str] = pydantic.Field(
        default_factory=list,
        description="Chronological list of messages in a conversation session")


class SemanticMemoryEntry(pydantic.BaseModel):
    """A single, self-contained unit of semantic memory."""
    name: str = pydantic.Field(
        description="A unique, descriptive name or title for the memory entry."
    )
    summary: str = pydantic.Field(
        description="A brief, one-sentence summary of the memory.")
    details: str = pydantic.Field(
        description="Comprehensive details about the memory.")
    source: str = pydantic.Field(
        description=
        "The origin of the information (e.g., 'user_conversation', 'document_ingestion')."
    )
    category_path: str = pydantic.Field(
        description=
        "A slash-delimited path for organizing the memory (e.g., 'projects/kwazii/schemas')."
    )
    old_entry_id: Optional[str] = pydantic.Field(
        default=None,
        description=
        "If this entry is an update, the ID of the entry it replaces.")


class SemanticMemoryEntryEntity(schemas.EventEntity):
    """A semantic memory entry with timestamps."""
    memory_entry: SemanticMemoryEntry = pydantic.Field(
        description="The semantic memory data.")


class SemanticMemoryOperation(pydantic.BaseModel):
    """Defines an operation to be performed on the semantic memory."""
    operation: Literal["add", "update", "delete"] = pydantic.Field(
        description="The type of memory operation to perform.")
    reasoning: str = pydantic.Field(
        description="The rationale for performing this memory operation.")
    entry_id: str = pydantic.Field(
        description="The unique identifier for the memory entry being targeted."
    )
    # The following fields are used for 'add' or 'update' operations
    name: Optional[str] = None
    summary: Optional[str] = None
    details: Optional[str] = None
    source: Optional[str] = None
    category_path: Optional[str] = None
    old_entry_id: Optional[str] = pydantic.Field(
        default=None,
        description=
        "For 'update' operations, the ID of the entry being replaced.")


class SemanticMemoryOperationEntity(schemas.EventEntity):
    """A semantic memory operation with timestamps."""
    memory_op: SemanticMemoryOperation = pydantic.Field(
        description="The semantic memory operation data.")
