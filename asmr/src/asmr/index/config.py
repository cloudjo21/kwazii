from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TokenizerType(Enum):
    SPLIT = "split"
    HF_AUTO = "hf_auto"
    MORPH = "morph"


class RepresentationType(Enum):
    SPARSE = "sparse"
    DENSE = "dense"


@dataclass
class FieldConfig:
    """Configuration for field indices"""
    name: str
    tokenizer_type: TokenizerType
    representation_type: RepresentationType
    model_path: Optional[str] = None  # Required for HF_AUTO tokenizers
    faiss_index_path: Optional[str] = None  # Required for dense fields
    
    def __post_init__(self):
        if self.tokenizer_type == TokenizerType.HF_AUTO and self.model_path is None:
            raise ValueError("model_path is required when tokenizer_type is HF_AUTO")
        
        if self.representation_type == RepresentationType.DENSE and self.faiss_index_path is None:
            raise ValueError("faiss_index_path is required when representation_type is DENSE")