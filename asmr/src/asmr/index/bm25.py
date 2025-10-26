import json
import os
from typing import Iterable

import numpy as np
from scipy.sparse import csc_matrix

from asmr import columnar
from asmr import vocab
from asmr.tokenize import helpers


# _INDEX_DIR = "/path/to/index/dir"
_INDEX_DIR = "./path/to/index/dir"
_INDEX_FILE = "indices.bin"
_DATA_FILE  = "data.bin"
_INDEX_POINTER_FILE = "indptr.bin"
_METADATA_FILE = "metadata.json"


class TermScore:
    term: str
    score: float

class DocumentScore:
    field_name: str
    doc_id: int
    term_scores: list[TermScore]

class BM25Index:
    field_name: str
    index: csc_matrix
    vocab: vocab.Vocabulary

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
        return DocumentScore(doc_id, term_scores)


class BM25Indexer:
    @classmethod
    def build(cls, field_name: str, columnar_posting: Iterable[str], vocab: vocab.Vocabulary, field_statistics: columnar.ColumnarStatistics) -> BM25Index:

        os.makedirs(_INDEX_DIR, exist_ok=True)
        index_filepath = os.path.join(_INDEX_DIR, _INDEX_FILE)
        data_filepath  =  os.path.join(_INDEX_DIR, _DATA_FILE)
        pointer_filepath = os.path.join(_INDEX_DIR, _INDEX_POINTER_FILE)
        metadata_filepath = os.path.join(_INDEX_DIR, _METADATA_FILE)

        k1, b = 1.5, 0.75  # BM25 parameters
        n_docs  = field_statistics.n_docs
        avg_doc_len = field_statistics.avg_doc_len
        df = field_statistics.df
        idf_of_terms = field_statistics.idf
        doc_len_of_docs = field_statistics.doc_len
        nnz_total = field_statistics.nnz_total
        n_vocab = len(vocab)

        indptr = np.empty(n_vocab + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(df, out=indptr[1:])

        indices = np.memmap(index_filepath, dtype=np.int32, mode="w+", shape=(nnz_total,))
        data = np.memmap(data_filepath,  dtype=np.float32, mode="w+", shape=(nnz_total,))
        offset = indptr.copy()

        text_stream = columnar.ColumnarTextStream(
            columnar_posting,
            vocab.trie,
            helpers.TokenizerWrapper(helpers.SplitTokenizer())
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

        # Save index pointer array
        np.save(pointer_filepath, indptr)

        # Save metadata if needed
        metadata = {
            "field_name": field_name,
            "n_docs": n_docs,
            "n_vocab": n_vocab,
            "avg_doc_len": avg_doc_len,
            "k1": k1,
            "b": b
        }
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f)

        bm25_index = BM25Index()
        bm25_index.field_name = field_name
        bm25_index.index = index
        bm25_index.vocab = vocab
        return bm25_index

    @classmethod
    def load(cls, field_name: str, vocab: vocab.Vocabulary) -> BM25Index:
        """Load a previously built BM25 index from disk files."""
        index_filepath = os.path.join(_INDEX_DIR, _INDEX_FILE)
        data_filepath  = os.path.join(_INDEX_DIR, _DATA_FILE)
        pointer_filepath = os.path.join(_INDEX_DIR, _INDEX_POINTER_FILE)
        metadata_filepath = os.path.join(_INDEX_DIR, _METADATA_FILE)

        # Check if all required files exist
        if not all(os.path.exists(path) for path in [index_filepath, data_filepath, pointer_filepath, metadata_filepath]):
            raise FileNotFoundError("Index files not found. Please build the index first.")

        with open(metadata_filepath, "r") as f:
            metadata = json.load(f)
            if metadata["field_name"] != field_name:
                raise ValueError(f"Field name mismatch: expected {metadata['field_name']}, got {field_name}")

        # Load index pointer array
        indptr = np.load(pointer_filepath)

        n_docs = int(metadata["n_docs"])
        n_vocab = len(vocab)  # equal to indptr.shape[0] - 1

        indices = np.memmap(index_filepath, dtype=np.int32, mode="r", shape=(indptr[-1],))
        data    = np.memmap(data_filepath,  dtype=np.float32, mode="r", shape=(indptr[-1],))

        # Reconstruct CSC matrix
        index = csc_matrix((data, indices, indptr), shape=(n_docs, n_vocab))

        bm25_index = BM25Index()
        bm25_index.field_name = field_name
        bm25_index.index = index
        bm25_index.vocab = vocab
        return bm25_index
