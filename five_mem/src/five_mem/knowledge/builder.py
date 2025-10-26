from typing import List
from datetime import datetime

from five_mem.knowledge import schemas


class SemanticMemoryEntryEntityBuilder:
    """Builds SemanticMemoryEntryEntity objects."""

    def build(
        self, entries: List[schemas.SemanticMemoryEntry]
    ) -> List[schemas.SemanticMemoryEntryEntity]:
        """
        Builds a list of SemanticMemoryEntryEntity from a list of SemanticMemoryEntry.

        Args:
            entries: A list of SemanticMemoryEntry objects.

        Returns:
            A list of SemanticMemoryEntryEntity objects, each with timestamps.
        """
        now = datetime.now()
        entities = []
        for entry in entries:
            entity = schemas.SemanticMemoryEntryEntity(created_at=now,
                                                       updated_at=now,
                                                       memory_entry=entry)
            entities.append(entity)
        return entities
