"""Text encoder protocols aligned with BaseFdeEncoder.encode_text."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

import numpy as np
import numpy.typing as npt
from torch import Tensor, nn

if TYPE_CHECKING:
    from fde.config import PromptType


@runtime_checkable
class TextEncoderProtocol(Protocol):
    """Fixed-dim text embedding for retrieval (BaseFdeEncoder-aligned)."""

    @property
    def embedding_dim(self) -> int: ...

    def encode_text(
        self,
        texts: str | Sequence[str],
        prompt_type: PromptType,
        *,
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        """Return float32 array [B, embedding_dim]. Inference-only (no grad)."""
        ...


@runtime_checkable
class NamedTextEncoderProtocol(TextEncoderProtocol, Protocol):
    """Text encoder with a stable cache-key slug."""

    @property
    def name(self) -> str: ...


@runtime_checkable
class TrainableTextEncoderProtocol(NamedTextEncoderProtocol, Protocol):
    """Gradient-enabled encoder for MFAR Phase 2 joint training."""

    def encode_trainable(
        self,
        texts: Sequence[str],
        prompt_type: PromptType,
        *,
        batch_size: int = 32,
    ) -> Tensor:
        """Return float32 tensor [B, D] on compute device with grad enabled."""
        ...

    def trainable_module(self) -> nn.Module:
        """Underlying module for optimizer param group."""
        ...

    def supports_joint_training(self) -> bool: ...


@runtime_checkable
class LoraTrainableTextEncoderProtocol(TrainableTextEncoderProtocol, Protocol):
    """LoRA-only trainable Jina encoder extension."""

    def lora_state_dict(self) -> dict[str, Tensor]: ...

    def load_lora_state_dict(self, state: dict[str, Tensor]) -> None: ...

    @property
    def lora_task(self) -> str: ...


@runtime_checkable
class MultiModalEncoderProtocol(TextEncoderProtocol, Protocol):
    """Optional image encoding for FDE encoders."""

    def encode_image(
        self,
        images: str | object | Sequence[str | object],
        *,
        batch_size: int = 8,
    ) -> npt.NDArray[np.float32]: ...


@runtime_checkable
class QueryEncoderProtocol(NamedTextEncoderProtocol, Protocol):
    """Benchmark encoder: canonical encode_text + legacy query/passage helpers."""

    def encode_passages(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        """Return float32 array [B, H] for document/passage texts."""
        ...

    def encode_queries(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> Tensor:
        """Return float32 tensor [B, H] for query texts."""
        ...

    def supports_joint_training(self) -> bool: ...


def encode_text_as_tensor(
    encoder: TextEncoderProtocol,
    texts: str | Sequence[str],
    prompt_type: PromptType,
    *,
    batch_size: int = 32,
) -> Tensor:
    """Legacy tensor adapter over canonical encode_text."""
    import torch

    arr = encoder.encode_text(
        texts,
        prompt_type,
        batch_size=batch_size,
    )
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))
