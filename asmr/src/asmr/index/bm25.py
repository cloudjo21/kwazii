import enum
import json
import os
from typing import Iterable

import numpy as np
from scipy.sparse import csc_matrix
import marisa_trie

from asmr import columnar
from asmr import vocab
from asmr.tokenize import helpers

# _INDEX_DIR = "/path/to/index/dir"
_INDEX_DIR = "./path/to/index/dir"
_INDEX_FILE = "indices.bin"
_DATA_FILE = "data.bin"
_INDEX_POINTER_FILE = "indptr.bin"
_METADATA_FILE = "metadata.json"


class TermScore:
    def __init__(self, term: str, score: float):
        self.term = term
        self.score = score


class DocumentScore:
    def __init__(self, field_name: str, doc_id: int, term_scores: list[TermScore]):
        self.field_name = field_name
        self.doc_id = doc_id
        self.term_scores = term_scores


class DocIdPostingPolicy(enum.Enum):
    UNIQUE = 1
    FIRST_WIN = 2
    LAST_WIN = 3


class DocumentIndexToIdMapping:
    def __init__(
        self,
        mapping: marisa_trie.BytesTrie,
        posting_policy: DocIdPostingPolicy = DocIdPostingPolicy.UNIQUE,
    ):
        self.mapping = mapping
        self.posting_policy = posting_policy

    def get_doc_id(self, doc_pos: int) -> str:
        """Get string doc_id for given integer doc_pos"""
        key = str(doc_pos)
        value = self.mapping.get(key)
        if value is not None:
            # access the first value of bytes-trie value tuple, bytes-tuple allow multiple values per key
            return value[0].decode("utf-8")
        else:
            raise KeyError(f"DOC_POSITION {doc_pos} not found in mapping.")

    @classmethod
    def build(
        cls,
        pairs: Iterable[tuple[int, str]],
        posting_policy: DocIdPostingPolicy = DocIdPostingPolicy.UNIQUE,
    ) -> "DocumentIndexToIdMapping":
        if posting_policy == DocIdPostingPolicy.UNIQUE:
            seen = set()

            def unique_pairs():
                for k, v in pairs:
                    if k in seen:
                        raise ValueError(f"duplicate key: {k!r}")
                    seen.add(k)
                    yield (str(k), v.encode("utf-8"))

            return cls(
                marisa_trie.BytesTrie(unique_pairs()), posting_policy=posting_policy
            )

        elif posting_policy == DocIdPostingPolicy.FIRST_WIN:
            seen = set()

            def first_win_pairs():
                for k, v in pairs:
                    if k not in seen:
                        seen.add(k)
                        yield (str(k), v.encode("utf-8"))

            return cls(
                marisa_trie.BytesTrie(first_win_pairs()), posting_policy=posting_policy
            )

        elif posting_policy == DocIdPostingPolicy.LAST_WIN:
            return cls(
                marisa_trie.BytesTrie((str(k), v.encode("utf-8")) for k, v in pairs),
                posting_policy=posting_policy,
            )
        else:
            raise ValueError(f"Unknown posting policy: {posting_policy}")


class BM25Index:
    def __init__(
        self,
        field_name: str,
        index: csc_matrix,
        vocab: vocab.Vocabulary,
        doc_id_mapping: DocumentIndexToIdMapping = None,
    ):
        self.field_name = field_name
        self.index = index
        self.vocab = vocab
        self.doc_id_mapping = doc_id_mapping

    def get_score(self, doc_id: int, terms: list[str]) -> DocumentScore:
        if not self.vocab or not self.vocab.trie:
            raise ValueError("Vocabulary is not set in BM25Index.")

        trie = self.vocab.trie

        if doc_id < 0 or doc_id >= self.index.shape[0]:
            raise ValueError(f"Document ID {doc_id} not found in index.")

        term_scores = []
        for term in terms:
            if term in trie:
                score = float(self.index[doc_id, trie[term]])
                term_score = TermScore(term, score)
                term_scores.append(term_score)
        return DocumentScore(self.field_name, doc_id, term_scores)


class BM25Indexer:
    @classmethod
    def build(
        cls,
        field_name: str,
        columnar_posting: Iterable[str],
        vocab: vocab.Vocabulary,
        field_statistics: columnar.ColumnarStatistics,
        doc_ids: Iterable[str] = None,
    ) -> BM25Index:
        os.makedirs(_INDEX_DIR, exist_ok=True)
        index_filepath = os.path.join(_INDEX_DIR, _INDEX_FILE)
        data_filepath = os.path.join(_INDEX_DIR, _DATA_FILE)
        pointer_filepath = os.path.join(_INDEX_DIR, _INDEX_POINTER_FILE)
        metadata_filepath = os.path.join(_INDEX_DIR, _METADATA_FILE)

        k1, b = 1.5, 0.75  # BM25 parameters
        n_docs = field_statistics.n_docs
        avg_doc_len = field_statistics.avg_doc_len
        df = field_statistics.df
        idf_of_terms = field_statistics.idf
        doc_len_of_docs = field_statistics.doc_len
        nnz_total = field_statistics.nnz_total
        n_vocab = len(vocab)

        indptr = np.empty(n_vocab + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(df, out=indptr[1:])

        indices = np.memmap(
            index_filepath, dtype=np.int32, mode="w+", shape=(nnz_total,)
        )
        data = np.memmap(data_filepath, dtype=np.float32, mode="w+", shape=(nnz_total,))
        offset = indptr.copy()

        text_stream = columnar.ColumnarTextStream(
            columnar_posting,
            vocab.trie,
            helpers.TokenizerWrapper(helpers.SplitTokenizer()),
        )

        for d, terms in enumerate(text_stream):
            uniq, counts = np.unique(terms, return_counts=True)
            dl = float(doc_len_of_docs[d])
            tf = counts.astype(np.float32)

            for term, freq in zip(uniq.astype(np.int32), tf):
                idf = idf_of_terms[term]
                denom = freq + k1 * (1 - b + b * (dl / avg_doc_len))
                score = idf * (freq * (k1 + 1)) / denom

                pos = offset[term]
                indices[pos] = d
                data[pos] = score
                offset[term] += 1

        index = csc_matrix((data, indices, indptr), shape=(n_docs, n_vocab))

        # Create doc_id_mapping if doc_ids provided
        doc_id_mapping = None
        if doc_ids is not None:
            # Create mapping using DocumentIndexToIdMapping without converting to list
            doc_id_mapping = DocumentIndexToIdMapping.build(
                enumerate(doc_ids), DocIdPostingPolicy.UNIQUE
            )

            # Note: We can't validate the length beforehand without converting to list
            # The validation will happen implicitly if there's a mismatch during index usage

        # Save index pointer array
        np.save(pointer_filepath, indptr)

        # Save metadata if needed
        metadata = {
            "field_name": field_name,
            "n_docs": n_docs,
            "n_vocab": n_vocab,
            "avg_doc_len": avg_doc_len,
            "k1": k1,
            "b": b,
        }
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f)

        return BM25Index(
            field_name=field_name,
            index=index,
            vocab=vocab,
            doc_id_mapping=doc_id_mapping,
        )

    @classmethod
    def load(cls, field_name: str, vocab: vocab.Vocabulary) -> BM25Index:
        """Load a previously built BM25 index from disk files."""
        index_filepath = os.path.join(_INDEX_DIR, _INDEX_FILE)
        data_filepath = os.path.join(_INDEX_DIR, _DATA_FILE)
        pointer_filepath = os.path.join(_INDEX_DIR, _INDEX_POINTER_FILE)
        metadata_filepath = os.path.join(_INDEX_DIR, _METADATA_FILE)

        # Check if all required files exist
        if not all(
            os.path.exists(path)
            for path in [
                index_filepath,
                data_filepath,
                pointer_filepath,
                metadata_filepath,
            ]
        ):
            raise FileNotFoundError(
                "Index files not found. Please build the index first."
            )

        with open(metadata_filepath, "r") as f:
            metadata = json.load(f)
            if metadata["field_name"] != field_name:
                raise ValueError(
                    f"Field name mismatch: expected {metadata['field_name']}, got {field_name}"
                )

        # Load index pointer array
        indptr = np.load(pointer_filepath)

        n_docs = int(metadata["n_docs"])
        n_vocab = len(vocab)  # equal to indptr.shape[0] - 1

        indices = np.memmap(
            index_filepath, dtype=np.int32, mode="r", shape=(indptr[-1],)
        )
        data = np.memmap(data_filepath, dtype=np.float32, mode="r", shape=(indptr[-1],))

        # Reconstruct CSC matrix
        index = csc_matrix((data, indices, indptr), shape=(n_docs, n_vocab))

        # Note: doc_id_mapping is not persisted in current implementation
        # It would need to be stored and loaded if persistence is required
        return BM25Index(
            field_name=field_name, index=index, vocab=vocab, doc_id_mapping=None
        )
