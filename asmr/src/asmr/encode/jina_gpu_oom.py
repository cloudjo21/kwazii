"""CUDA OOM detection and recovery for Jina FDE index encoding."""

import logging

import torch

logger = logging.getLogger(__name__)

JINA_DEFAULT_ENCODE_BATCH = 8
JINA_OOM_ENCODE_BATCH = 4


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True when *exc* looks like a CUDA out-of-memory failure."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def clear_cuda_cache() -> None:
    """Best-effort release of cached CUDA allocations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
