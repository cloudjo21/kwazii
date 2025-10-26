from datetime import datetime
from typing import List, Optional, Literal
from uuid import UUID

import pydantic

from five_mem import schemas


class EventBoundaryTurn(pydantic.BaseModel):
    """Event boundary detection result for a specific turn"""
    turn: int = pydantic.Field(
        description="Turn number where the boundary was detected"
    )
    boundary_type: List[Literal[
        "context_change", 
        "topic_transition", 
        "goal_modification", 
        "emotion_change", 
        "surprise_occurrence"
    ]] = pydantic.Field(
        description="Types of boundaries detected at this turn"
    )
    evidence: str = pydantic.Field(
        description="Brief rationale for boundary detection in 1-3 sentences"
    )
    summary: str = pydantic.Field(
        description="Event summary in 1-3 sentences"
    )


class EventMessages(schemas.EventEntity):
    """Collection of messages that form an event"""
    id: UUID = pydantic.Field(
        description="Unique identifier for the event messages collection"
    )
    session_id: UUID = pydantic.Field(
        description="Identifier of the conversation session"
    )
    messages: List[schemas.Message] = pydantic.Field(
        description="List of messages that comprise this event"
    )
    begin_of_turn: int = pydantic.Field(
        description="Starting turn number of the event"
    )
    end_of_turn: int = pydantic.Field(
        description="Ending turn number of the event"
    )


class EventBoundary(schemas.EventEntity):
    """Event boundary marker"""
    id: UUID = pydantic.Field(
        description="Unique identifier for the event boundary"
    )
    boundary_turn: EventBoundaryTurn = pydantic.Field(
        description="Boundary detection information for this event"
    )


class EventCandidate(schemas.EventEntity):
    """Candidate event for boundary detection"""
    id: UUID = pydantic.Field(
        description="Unique identifier for the event candidate"
    )
    event_msg_id: UUID = pydantic.Field(
        description="Reference to the EventMessages that this candidate is based on"
    )
    boundary: Optional[EventBoundary] = pydantic.Field(
        default=None,
        description="Detected boundary information (None if no boundary detected)"
    )

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
