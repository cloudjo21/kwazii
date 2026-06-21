"""Query encoders for benchmark evaluation.

Implementations moved to asmr.encode.implementations and asmr.encode.adapters.
This module re-exports all public symbols and hosts the create_query_encoder() factory.
"""

import logging
import os
from pathlib import Path

import torch

from asmr.encode.adapters import (
    HfQueryEncoderAdapter,
    _JinaFdeTextAdapter,
    _JinaLoraEncoderWrapper,
    _validate_mrl_dim,
)
from asmr.encode.implementations import (
    HfQueryEncoder,
    JinaLoraConfig,
    JinaLoraQueryEncoder,
    JinaQueryEncoder,
)
from asmr.encode.protocol import (
    QueryEncoderProtocol,
)
from asmr.encode.jina_gpu_oom import (
    JINA_DEFAULT_ENCODE_BATCH,
    JINA_OOM_ENCODE_BATCH,
    clear_cuda_cache,
    is_cuda_oom,
)

_DEFAULT_JINA_MODEL = os.getenv(
    "ENCODER_MODEL_PATH",
    "jinaai/jina-embeddings-v4",
)

logger = logging.getLogger(__name__)


def create_query_encoder(
    name: str = "jinavera",
    *,
    model_path: str | None = None,
    hf_model_name: str = "facebook/contriever-msmarco",
    truncate_dim: int = 1024,
    fde_output_dim: int | None = 1024,
    device: str | None = None,
    for_training: bool = False,
    lora_adapter_name: str = "stark_prime",
    lora_checkpoint: Path | str | None = None,
) -> QueryEncoderProtocol:
    """Factory for benchmark query encoders.

    Default inference: frozen JinaQueryEncoder (Jinavera FDE).
    Training: JinaLoraQueryEncoder when name is jinavera-lora / for_training.
    """
    key = name.lower().strip()
    if key in ("jinavera", "jinavera-lora", "jina", "jina-lora", "jinavera1024"):
        _validate_mrl_dim(truncate_dim, label="truncate_dim")
    if for_training and key in ("jinavera", "jinavera-lora", "jina", "jina-lora"):
        cfg = JinaLoraConfig(
            model_path=model_path or _DEFAULT_JINA_MODEL,
            truncate_dim=truncate_dim,
            adapter_name=lora_adapter_name,
        )
        return _JinaLoraEncoderWrapper(
            JinaLoraQueryEncoder(cfg, device=device),
        )
    if key in ("jinavera-lora", "jina-lora") and not for_training:
        cfg = JinaLoraConfig(
            model_path=model_path or _DEFAULT_JINA_MODEL,
            truncate_dim=truncate_dim,
            adapter_name=lora_adapter_name,
        )
        enc = JinaLoraQueryEncoder(cfg, device=device)
        if lora_checkpoint is not None:
            enc.load_lora_state_dict(
                torch.load(lora_checkpoint, weights_only=True),
            )
        return _JinaLoraEncoderWrapper(enc)
    if key in ("jinavera", "jina", "jinavera1024"):
        return JinaQueryEncoder(
            model_path=model_path,
            truncate_dim=truncate_dim,
            fde_output_dim=fde_output_dim,
            lora_checkpoint=lora_checkpoint,
        )
    if key in ("contriever", "hf", "huggingface"):
        return HfQueryEncoderAdapter(model_name=hf_model_name, device=device)
    raise ValueError(f"Unknown encoder: {name!r}")


__all__ = [
    "QueryEncoderProtocol",
    "HfQueryEncoder",
    "HfQueryEncoderAdapter",
    "JinaLoraConfig",
    "JinaLoraQueryEncoder",
    "JinaQueryEncoder",
    "_JinaFdeTextAdapter",
    "_JinaLoraEncoderWrapper",
    "create_query_encoder",
    # OOM utils (legacy callers)
    "JINA_DEFAULT_ENCODE_BATCH",
    "JINA_OOM_ENCODE_BATCH",
    "clear_cuda_cache",
    "is_cuda_oom",
]
