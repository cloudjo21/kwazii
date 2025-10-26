from datetime import datetime
from typing import List, Optional, Literal
from uuid import UUID, uuid4

import pydantic

from five_mem import schemas as base_schemas
from five_mem.boundary import schemas as boundary_schemas


class EpisodeCandidate(pydantic.BaseModel):
# class EpisodeCandidate(boundary_schemas.EventEntity):
    """A candidate episode that combines message history and boundary information.
    
    This model represents a potential episode by combining:
    1. Message history from EventMessages (formatted as 'author: message')
    2. Boundary information from EventBoundaryTurn if present
    """
    # id: UUID = pydantic.Field(
    #     description="Unique identifier for the episode candidate"
    # )
    # event_messages_id: UUID = pydantic.Field(
    #     description="Reference to the source EventMessages"
    # )
    message_history: str = pydantic.Field(
        description="Concatenated messages in 'author: payload' format, separated by newlines"
    )
    begin_of_turn: int = pydantic.Field(
        description="Starting turn number in this episode"
    )
    end_of_turn: int = pydantic.Field(
        description="Ending turn number in this episode"
    )
    message_count: int = pydantic.Field(
        description="Total number of messages in this episode"
    )
    
    # Optional boundary information
    boundary_turn: int = pydantic.Field(
        description="Turn number where boundary was detected"
    )
    boundary_types: List[Literal[
        "context_change",
        "topic_transition",
        "goal_modification",
        "emotion_change",
        "surprise_occurrence"
    ]] = pydantic.Field(
        description="Types of boundaries detected at the boundary turn"
    )
    boundary_evidence: str = pydantic.Field(
        description="Evidence supporting boundary detection"
    )
    boundary_summary: str = pydantic.Field(
        description="Summary of the boundary event"
    )

    @classmethod
    def from_event_messages(cls, event_messages: boundary_schemas.EventMessages, boundary: boundary_schemas.EventBoundaryTurn) -> "EpisodeCandidate":
        """Create EpisodeCandidate from EventMessages and optional EventBoundaryTurn"""
        # Format message history
        message_history = "\n".join(
            f"{msg.author}: {msg.payload}" 
            for msg in event_messages.messages
        )
        
        # Create base candidate
        candidate = cls(
            # id=uuid4(),
            # event_messages_id=event_messages.id,
            message_history=message_history,
            begin_of_turn=event_messages.begin_of_turn,
            end_of_turn=event_messages.end_of_turn,
            message_count=len(event_messages.messages),
            boundary_turn=boundary.turn,
            boundary_types=boundary.boundary_type,
            boundary_evidence=boundary.evidence,
            boundary_summary=boundary.summary,
            # created_at=datetime.utcnow(),
            # updated_at=datetime.utcnow()
        )
        
        return candidate

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Insight(pydantic.BaseModel):
    """Insight analysis derived from an Episode object.
    
    This model captures key insights such as constraints, decisions, and follow-ups
    identified from the episode context.
    """
    constraints: List[str] = pydantic.Field(
        description="Essential constraints (budget, dates, requirements, etc.) - 1 to 5 items"
    )
    decisions: List[str] = pydantic.Field(
        description="Key decisions or agreements made - 1 to 3 items"
    )
    followups: List[str] = pydantic.Field(
        description="Items to continue in the next session - 1 to 3 items"
    )


class Episode(pydantic.BaseModel):
    """
    Event boundary episode that captures context changes in conversation.
    Represents points where context shifts dramatically (situation/emotion/topic/goal/surprise).
    """

    title: str = pydantic.Field(
        description="Episode title (3-7 words describing the main event)"
    )
    summary: str = pydantic.Field(
        description="Episode summary in 3-5 sentences capturing the key narrative"
    )
    
    # A. Context-State
    who: str = pydantic.Field(
        description="Speaker or participant in the conversation"
    )
    where: str = pydantic.Field(
        description="Location or domain context (e.g., 'travel planning', 'work tasks', 'family health')"
    )
    when: datetime = pydantic.Field(
        description="Relative or absolute timestamp within the session"
    )
    
    # B. Topic
    topic: str = pydantic.Field(
        description="High-level task or subject matter"
    )
    subtopic: Optional[str] = pydantic.Field(
        default=None,
        description="Specific sub-task or detailed question within the main topic"
    )
    
    # C. Goal/Intent
    user_goal: str = pydantic.Field(
        description="User's objective to achieve (problem solving, planning, decision making, etc.) expressed with domain keywords"
    )
    agent_commitments: list[str] = pydantic.Field(
        default_factory=list,
        description="Actions or outputs the agent has committed to provide (e.g., 'provide code snippet', 'create travel itinerary draft')"
    )
    
    # D. Affect
    affective_cue: Optional[str] = pydantic.Field(
        default=None,
        description="Emotional text, emoji, or adjective hints related to sentiment"
    )
    salient_affect: Optional[str] = pydantic.Field(
        default=None,
        description="Particularly emphasized emotions (anger, sadness, excitement, etc.)"
    )
    
    # E. Surprise/Anomaly
    surprise_event: Optional[str] = pydantic.Field(
        default=None,
        description="Situation where there's a gap between prediction and actual conversation development (new information, sudden requirement changes)"
    )
    policy_flags: Optional[List[str]] = pydantic.Field(
        default=None,
        description="Sensitive/security/prohibited topic transitions (safety guardrail signals)"
    )
    
    # G. Boundary Meta
    boundary_type: Literal[
        "context_change", 
        "topic_transition", 
        "goal_modification", 
        "emotion_change", 
        "surprise_occurrence"
    ] = pydantic.Field(
        description="Type of boundary that triggered this episode"
    )
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "who": "user",
                "where": "travel planning",
                "when": "2025-10-18T10:30:00",
                "topic": "vacation itinerary",
                "subtopic": "restaurant recommendations",
                "user_goal": "find family-friendly restaurants in Seoul with budget constraints",
                "agent_commitments": ["provide restaurant list", "include price ranges"],
                "affective_cue": "excited emoji 😊",
                "salient_affect": "excitement",
                "surprise_event": None,
                "policy_flags": None,
                "boundary_type": "topic_transition",
            }
        }


class EpisodeAfterBoundary(Episode):

    # H. Insight
    constraints: list[str] = pydantic.Field(
        default_factory=list,
        max_items=5,
        description="Essential constraints (budget, dates, requirements, etc.) - 1 to 5 items"
    )
    decisions: list[str] = pydantic.Field(
        default_factory=list,
        max_items=3,
        description="Key decisions or agreements made - 1 to 3 items"
    )
    followups: list[str] = pydantic.Field(
        default_factory=list,
        max_items=3,
        description="Items to continue in the next session - 1 to 3 items"
    )

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "title": "Family trip to Seoul",
                "summary": "Planning a family vacation to Seoul with a focus on kid-friendly activities and dining options.",
                "who": "user",
                "where": "travel planning",
                "when": "2025-10-18T10:30:00",
                "topic": "vacation itinerary",
                "subtopic": "restaurant recommendations",
                "user_goal": "find family-friendly restaurants in Seoul with budget constraints",
                "agent_commitments": ["provide restaurant list", "include price ranges"],
                "affective_cue": "excited emoji 😊",
                "salient_affect": "excitement",
                "surprise_event": None,
                "policy_flags": None,
                "boundary_type": "topic_transition",
                "constraints": ["budget under $50 per meal", "kid-friendly venues"],
                "decisions": ["focus on Gangnam district"],
                "followups": ["book reservations", "check opening hours"]
            }
        }


class EpisodeEntity(base_schemas.EventEntity):
    """Episode entity with unique identifier and timestamps"""
    episode: EpisodeAfterBoundary = pydantic.Field(
        description="The episode data"
    )