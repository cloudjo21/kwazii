"""GPU device setup with CUDA MPS thread percentage support.

Call setup_gpu() before any torch.cuda usage — CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
must be set before the first CUDA context is created.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_MPS_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"


def setup_gpu(mps_thread_pct: int | None = None) -> torch.device:
    """Select the best available device and optionally configure CUDA MPS.

    Args:
        mps_thread_pct: If set and the env var is not already present, writes
            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE before the CUDA context is
            created. Ignored on non-CUDA platforms.

    Returns:
        Selected torch.device (cuda / mps / cpu).
    """
    if mps_thread_pct is not None and _MPS_ENV not in os.environ:
        os.environ[_MPS_ENV] = str(mps_thread_pct)

    pct = os.environ.get(_MPS_ENV)
    if pct:
        logger.info("CUDA MPS thread percentage: %s%%", pct)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
