
from typing import Dict, Any
from uuid import uuid4
from datetime import datetime

from five_mem.chunk.schemas import Insight
from five_mem.schemas import Episode, EpisodeAfterBoundary, EpisodeEntity

# Constants
EPISODE_LOGS = "episode_logs"


class EpisodeStateUpdater:
    """Updates episode state in context with new Episode objects"""
    
    def __init__(self, states: Dict[str, Any]):
        """Initialize with context states dictionary
        
        Args:
            states: Context states dictionary containing episodic memory
        """
        self.states = states
        
    def update(self, episode: Episode, user_id: str) -> None:
        """Update episode logs in context state with new Episode
        
        Args:
            episode: Episode model object from LLM using EPISODIC_MEMORY_COMPRESSOR_PROMPT
            user_id: ID of the user associated with this episode
        """
        # Initialize episode_logs if not exists
        if EPISODE_LOGS not in self.states:
            self.states[EPISODE_LOGS] = {}
            
        # Initialize user logs if not exists
        user_key = f"user:{user_id}"
        if user_key not in self.states[EPISODE_LOGS]:
            self.states[EPISODE_LOGS][user_key] = []
            
        # Add episode to logs
        self.states[EPISODE_LOGS][user_key].append(episode)


class EpisodeEntityBuilder:
    """Builds EpisodeEntity objects from Episode and Insight data"""
    
    def build(self, episode: Episode, insight: Insight) -> EpisodeEntity:
        """Create EpisodeEntity by combining Episode with Insight
        Args:
            episode: Base Episode object with context, topics, goals
            insight: Insight object with constraints, decisions, followups
            
        Returns:
            EpisodeAfterBoundary with combined fields
        """
        # Create EpisodeAfterBoundary by extending Episode with Insight fields
        episode_after_boundary = EpisodeAfterBoundary(
            # Copy all base Episode fields
            title=episode.title,
            summary=episode.summary,
            who=episode.who,
            where=episode.where,
            when=episode.when,
            topic=episode.topic,
            subtopic=episode.subtopic,
            user_goal=episode.user_goal,
            agent_commitments=episode.agent_commitments,
            affective_cue=episode.affective_cue,
            salient_affect=episode.salient_affect,
            surprise_event=episode.surprise_event,
            policy_flags=episode.policy_flags,
            boundary_type=episode.boundary_type,
            
            # Add Insight fields
            constraints=insight.constraints,
            decisions=insight.decisions,
            followups=insight.followups
        )

        episode_entity = EpisodeEntity(
            id=uuid4(),
            episode=episode_after_boundary,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            deleted_at=None
        )

        return episode_entity


class EpisodeAfterBoundaryStateUpdater:
    """Clears episode logs after boundary analysis stage"""
    
    def __init__(self, states: Dict[str, Any]):
        """Initialize with context states dictionary
        
        Args:
            states: Context states dictionary containing episodic memory
        """
        self.states = states
        
    def update(self, user_id: str) -> None:
        """Clear episode logs for a specific user after LLM analysis
        
        Args:
            user_id: ID of the user whose logs should be cleared
        """
        user_key = f"user:{user_id}"
        
        if EPISODE_LOGS in self.states and user_key in self.states[EPISODE_LOGS]:
            # Clear the episode logs for this user
            self.states[EPISODE_LOGS][user_key] = []


# Example usage:
def process_episode(
    states: Dict[str, Any],
    episode: Episode,
    insight: Insight,
    user_id: str
) -> EpisodeEntity:
    """
    Complete pipeline for processing an episode
    
    Args:
        states: Context states dictionary
        episode: Episode object from LLM
        insight: Insight object from LLM
        user_id: User identifier
        
    Returns:
        EpisodeAfterBoundary with combined data
    """
    # Update episode state
    episode_updater = EpisodeStateUpdater(states)
    episode_updater.update(episode, user_id)

    # Build EpisodeEntity
    builder = EpisodeEntityBuilder()
    episode_entity = builder.build(episode, insight)

    # Clear episode logs after processing
    after_boundary_updater = EpisodeAfterBoundaryStateUpdater(states)
    after_boundary_updater.update(user_id)

    return episode_entity
