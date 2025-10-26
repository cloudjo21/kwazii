from typing import Dict, List, Any
from uuid import uuid4, UUID
from datetime import datetime

from five_mem.boundary.schemas import (
    Message, EventMessages, EventCandidate, EventBoundary, 
    EventBoundaryTurn
)


class LastStateUpdater:
    """
    Updates conversation state after processing
    Manages conversation state by updating the current turn and clearing processed messages
    """
    
    def __init__(self, states: Dict[str, Any]):
        self.states = states
    
    def update(self, new_turn: int) -> None:
        """
        Update CURRENT_TURN and clear LAST_MESSAGES
        to apply the agent callbacks of google adk
        """
        self.states["CURRENT_TURN"] = new_turn
        self.states["LAST_MESSAGES"] = []


class EventMessagesBuilder:
    """
    Builds EventMessages from boundaries and conversation state
    Creates EventMessages by segmenting conversations based on detected boundaries
    """
    
    def __init__(self, states: Dict[str, Any]):
        self.states = states
    
    def build(self, event_boundaries: List[EventBoundaryTurn], session_id: UUID) -> List[EventMessages]:
        """
        Use EventBoundaries, CURRENT_TURN, and LAST_MESSAGES to build EventMessages
        """
        current_turn = self.states.get("CURRENT_TURN", 0)
        last_messages = self.states.get("LAST_MESSAGES", [])
        
        event_messages_list = []
        
        if not event_boundaries:
            # No boundaries detected, create single event with all messages
            if last_messages:
                event_msg = self._create_event_messages(
                    session_id=session_id,
                    messages=last_messages,
                    begin_turn=min(msg.turn for msg in last_messages),
                    end_turn=max(msg.turn for msg in last_messages)
                )
                event_messages_list.append(event_msg)
        else:
            # Group messages by boundaries
            boundary_turns = [b.turn for b in event_boundaries]
            boundary_turns.sort()
            
            # Add start and end boundaries for segmentation
            all_turns = [0] + boundary_turns + [current_turn]
            
            for i in range(len(all_turns) - 1):
                start_turn = all_turns[i]
                end_turn = all_turns[i + 1]
                
                # Get messages in this range
                segment_messages = [
                    msg for msg in last_messages 
                    if start_turn < msg.turn <= end_turn
                ]
                
                if segment_messages:
                    event_msg = self._create_event_messages(
                        session_id=session_id,
                        messages=segment_messages,
                        begin_turn=start_turn + 1,
                        end_turn=end_turn
                    )
                    event_messages_list.append(event_msg)
        
        return event_messages_list
    
    def _create_event_messages(
        self, 
        session_id: UUID, 
        messages: List[Message], 
        begin_turn: int, 
        end_turn: int
    ) -> EventMessages:
        """Create EventMessages object with timestamp metadata"""
        now = datetime.utcnow()
        
        return EventMessages(
            id=uuid4(),
            session_id=session_id,
            messages=messages,
            begin_of_turn=begin_turn,
            end_of_turn=end_turn,
            created_at=now,
            updated_at=now,
            deleted_at=None
        )


class EventCandidatesBuilder:
    """
    Builds EventCandidate objects from EventMessages
    Converts EventMessages into EventCandidate objects with optional boundary information
    """
    
    def __init__(self, states: Dict[str, Any]):
        self.states = states
    
    def build(
        self, 
        event_messages_list: List[EventMessages],
        event_boundaries: List[EventBoundaryTurn]
    ) -> List[EventCandidate]:
        """
        Use EventMessages to build several EventCandidate objects
        """
        event_candidates = []
        
        # Create boundary lookup for matching
        boundary_lookup = {b.turn: b for b in event_boundaries}
        
        for event_msg in event_messages_list:
            # Check if this event has a boundary at its end turn
            boundary = None
            if event_msg.end_of_turn in boundary_lookup:
                boundary_turn = boundary_lookup[event_msg.end_of_turn]
                boundary = self._create_event_boundary(boundary_turn)
            
            # Create event candidate
            candidate = self._create_event_candidate(
                event_msg_id=event_msg.id,
                boundary=boundary
            )
            event_candidates.append(candidate)
        
        return event_candidates
    
    def _create_event_boundary(self, boundary_turn: EventBoundaryTurn) -> EventBoundary:
        """Create EventBoundary from EventBoundaryTurn"""
        now = datetime.utcnow()
        
        return EventBoundary(
            id=uuid4(),
            boundary_turn=boundary_turn,
            created_at=now,
            updated_at=now,
            deleted_at=None
        )
    
    def _create_event_candidate(
        self, 
        event_msg_id: UUID, 
        boundary: EventBoundary = None
    ) -> EventCandidate:
        """Create EventCandidate object"""
        now = datetime.utcnow()
        
        return EventCandidate(
            id=uuid4(),
            event_msg_id=event_msg_id,
            boundary=boundary,
            created_at=now,
            updated_at=now,
            deleted_at=None
        )


# Usage example:
def process_event_boundaries(
    states: Dict[str, Any],
    event_boundaries: List[EventBoundaryTurn],
    session_id: UUID
) -> List[EventCandidate]:
    """
    Complete pipeline for processing event boundaries
    """
    # Build EventMessages
    event_messages_builder = EventMessagesBuilder(states)
    event_messages_list = event_messages_builder.build(event_boundaries, session_id)
    
    # Build EventCandidates
    candidates_builder = EventCandidatesBuilder(states)
    event_candidates = candidates_builder.build(event_messages_list, event_boundaries)
    
    # Update state for next processing
    current_turn = states.get("CURRENT_TURN", 0)
    state_updater = LastStateUpdater(states)
    state_updater.update(current_turn + 1)
    
    return event_candidates
