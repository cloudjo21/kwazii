import abc

from asmr.index import bm25
from asmr.tokenize import helpers


class BaseFieldIndex(abc.ABC):
    pass


class TextFieldIndex(BaseFieldIndexer):

    def __init__(self, index: bm25.BM25Index):
        self.tokenizer = helpers.TokenizerWrapper(helpers.SplitTokenizer())
        self.index = index


class DocumentIndex:
    def __init__(self):
        self.field_indices: dict[str, BaseFieldIndex] = dict()

    def add_field_index(self, field_name: str, index: BaseFieldIndex):
        self.field_indices[field_name] = index
