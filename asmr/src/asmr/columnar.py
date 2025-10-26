from typing import Iterable
import numpy as np
import marisa_trie
import pydantic

from asmr import vocab
from asmr.tokenize import helpers


# def stream_docs(columnar_texts: Iterable[str], token2id: marisa_trie.Trie) -> Iterable[np.ndarray]:
#     for text in columnar_texts:
#         tokens = text.lower().split()  # TODO: improve tokenization
#         token_ids = np.array([token2id[token] for token in tokens if token in token2id], dtype=np.int32)
#         yield token_ids

class ColumnarTextStream(Iterable):
    def __init__(self, texts: Iterable[str], token2id: marisa_trie.Trie, tokenizer: helpers.TokenizerWrapper):
        self.texts = texts
        self.token2id = token2id
        self.tokenizer = tokenizer

    def __iter__(self):
        for text in self.texts:
            tokens = text.lower().split()  # TODO: improve tokenization
            token_ids = np.array([self.token2id[token] for token in tokens if token in self.token2id], dtype=np.int32)
            yield token_ids


class FieldBasedColumnarTexts(Iterable):
    def __init__(self, documents: Iterable[dict], field_name: str, n_docs: int):
        self.documents = documents
        self.field_name = field_name
        self.n_docs = n_docs

    def __iter__(self):
        for doc in self.documents:
            yield doc[self.field_name]


class ColumnarStatistics(pydantic.BaseModel):
    field_name: str

    n_docs: int
    df: np.ndarray = pydantic.Field(
        default_factory=lambda: np.array([], dtype=np.int32))
    doc_len: np.ndarray = pydantic.Field(
        default_factory=lambda: np.array([], dtype=np.int32))
    idf: np.ndarray = pydantic.Field(
        default_factory=lambda: np.array([], dtype=np.float32))
    nnz_total: int = 0
    avg_doc_len: float = pydantic.Field(default=0.0)

    class Config:
        arbitrary_types_allowed = True


class ColumnarStatisticsBuilder:

    @classmethod
    def build(cls, field_columnar_texts: FieldBasedColumnarTexts, vocab: vocab.Vocabulary) -> ColumnarStatistics:
        if not vocab or not vocab.trie:
            raise ValueError("Vocabulary is not provided or invalid.")

        n_docs = field_columnar_texts.n_docs
        df = np.zeros(len(vocab), dtype=np.int32)
        doc_len = np.zeros(n_docs, dtype=np.int32)
        nnz_total = 0

        text_stream = ColumnarTextStream(
            field_columnar_texts,
            vocab.trie,
            helpers.TokenizerWrapper(helpers.SplitTokenizer())
        )

        for d, terms in enumerate(text_stream):
            doc_len[d] = len(terms)
            uniq = np.unique(terms)
            df[uniq] += 1
            nnz_total += uniq.size

            if d > n_docs-1:
                break

        idf = np.log((n_docs - df + 0.5) / (df + 0.5))
        idf = np.maximum(idf, 0)
        avg_doc_len = doc_len.mean() if n_docs > 0 else 0.0

        return ColumnarStatistics(
            field_name=field_columnar_texts.field_name,
            n_docs=n_docs,
            df=df,
            doc_len=doc_len,
            idf=idf,
            nnz_total=nnz_total,
            avg_doc_len=avg_doc_len
        )
