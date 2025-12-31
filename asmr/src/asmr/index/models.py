from typing import List
from pydantic import BaseModel


class FieldBasedRankingItem(BaseModel):
    """Individual ranking item with document ID and score"""
    doc_id: str
    score: float


class FieldBasedRanking(BaseModel):
    """Ranking results for a field-based search"""
    field_name: str
    query: str
    items: List[FieldBasedRankingItem]
    total_retrieved: int

    @property
    def doc_ids(self) -> List[str]:
        """Get list of document IDs in ranking order"""
        return [item.doc_id for item in self.items]

    @property
    def scores(self) -> List[float]:
        """Get list of scores in ranking order"""
        return [item.score for item in self.items]

    def top_k(self, k: int) -> "FieldBasedRanking":
        """Return top-k results"""
        return FieldBasedRanking(field_name=self.field_name,
                                 query=self.query,
                                 items=self.items[:k],
                                 total_retrieved=len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        """Make the class subscriptable to access items by index or slice"""
        if isinstance(index, slice):
            # Handle slicing - return a list of (doc_id, score) tuples
            return [(item.doc_id, item.score) for item in self.items[index]]
        else:
            # Handle single index access
            return (self.items[index].doc_id, self.items[index].score)

    def __iter__(self):
        """Make the class iterable, yielding (doc_id, score) tuples"""
        for item in self.items:
            yield (item.doc_id, item.score)

    def items(self) -> List[tuple[str, float]]:
        """Return list of (doc_id, score) tuples"""
        return [(item.doc_id, item.score) for item in self.items]
