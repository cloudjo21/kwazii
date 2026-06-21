"""Re-export shim — implementations moved to asmr.encode.jina_gpu_oom."""

from asmr.encode.jina_gpu_oom import (
    JINA_DEFAULT_ENCODE_BATCH,
    JINA_OOM_ENCODE_BATCH,
    clear_cuda_cache,
    is_cuda_oom,
)

__all__ = [
    "JINA_DEFAULT_ENCODE_BATCH",
    "JINA_OOM_ENCODE_BATCH",
    "clear_cuda_cache",
    "is_cuda_oom",
]
