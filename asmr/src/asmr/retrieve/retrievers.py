import abc

from asmr.index import bm25
from asmr.src.asmr.index import fields
from asmr.tokenize import helpers


TOP_K_RETRIEVE = 100


class BaseFieldRetriever(abc.ABC):
    @abc.abstractmethod
    def retrieve(self, field: str, query: str, k: int):
        pass


class TextFieldRetriever(BaseFieldRetriever):
    def __init__(self, document_index: fields.DocumentIndex):
        self.tokenizer = helpers.TokenizerWrapper(helpers.SplitTokenizer())
        self.field_indices: dict[str, fields.BaseFieldIndex] = document_index.field_indices

    def retrieve(self, field: str, query: str, k: int):
        if field not in self.field_indices:
            raise ValueError(f"Field '{field}' not found in TextFieldRetriever.")
        index = self.field_indices[field]
        if not isinstance(index, fields.TextFieldIndex):
            raise ValueError(f"Field '{field}' is not a TextFieldIndex.")
        terms = self.tokenizer.tokenize(query)
        scores = []
        for doc_id in range(index.index.shape[0]):
            doc_score = index.get_score(doc_id, terms)
            total_score = sum(ts.score for ts in doc_score.term_scores)
            if total_score > 0:
                scores.append((doc_id, total_score))
        # Sort by score descending and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)

        k = min(k, TOP_K_RETRIEVE)
        return scores[:k]


class FieldComplexRetriever:
    def __init__(self, field_retrievers: dict[str, BaseFieldRetriever]):
        self.field_retrievers = field_retrievers

    @property
    def fields(self) -> list[str]:
        return list(self.field_retrievers.keys())

    def retrieve(self, field: str, query: str, k: int):
        if field not in self.field_retrievers:
            raise ValueError(f"Field '{field}' not found in FieldComplexRetriever.")
        retriever = self.field_retrievers[field]
        return retriever.retrieve(field, query, k)
