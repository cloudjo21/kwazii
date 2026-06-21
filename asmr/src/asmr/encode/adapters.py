"""Adapter classes bridging encoder implementations to QueryEncoderProtocol."""

from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt
from torch import Tensor

from fde.config import MRL_DIMENSIONS, PromptType

from asmr.encode.implementations.hf import HfQueryEncoder
from asmr.encode.implementations.jina_lora import JinaLoraQueryEncoder
from asmr.encode.protocol import encode_text_as_tensor


def _apply_prefix_truncate(
    vectors: npt.NDArray[np.float32],
    dim: int,
    *,
    prefix_truncate: bool,
) -> npt.NDArray[np.float32]:
    """Slice and L2-normalize when legacy full-dim FDE exceeds target dim."""
    if not prefix_truncate or vectors.shape[1] <= dim:
        return vectors.astype(np.float32)
    out = vectors[:, :dim].astype(np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


def _validate_mrl_dim(dim: int, *, label: str) -> None:
    if dim not in MRL_DIMENSIONS:
        allowed = ", ".join(str(value) for value in MRL_DIMENSIONS)
        msg = f"{label}={dim} not in MRL dimensions ({allowed})"
        raise ValueError(msg)


def _resolve_jina_dims(
    truncate_dim: int,
    fde_output_dim: int | None,
) -> tuple[int, int | None]:
    """Return (mrl_dim, fde_output_dim) for Jina encoders."""
    _validate_mrl_dim(truncate_dim, label="truncate_dim")
    if fde_output_dim is not None:
        _validate_mrl_dim(fde_output_dim, label="fde_output_dim")
    return truncate_dim, fde_output_dim


class HfQueryEncoderAdapter:
    """Contriever-style HF encoder (legacy mFAR paper alignment)."""

    def __init__(
        self,
        model_name: str = "facebook/contriever-msmarco",
        *,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._encoder = HfQueryEncoder(model_name=model_name, device=device)

    @property
    def name(self) -> str:
        return self._encoder.name

    @property
    def embedding_dim(self) -> int:
        return self._encoder.embedding_dim

    def encode_text(
        self,
        texts: str | Sequence[str],
        prompt_type: PromptType = PromptType.QUERY,
        *,
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        return self._encoder.encode_text(
            texts,
            prompt_type,
            batch_size=batch_size,
        )

    def encode_passages(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        return self.encode_text(texts, PromptType.PASSAGE, batch_size=batch_size)

    def encode_queries(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> Tensor:
        return encode_text_as_tensor(
            self,
            texts,
            PromptType.QUERY,
            batch_size=batch_size,
        )

    def encode(self, texts: list[str]) -> Tensor:
        """Legacy SupportsQueryEncode compatibility."""
        return self.encode_queries(texts)

    def encode_trainable(
        self,
        texts: list[str],
        prompt_type: PromptType = PromptType.QUERY,
        *,
        batch_size: int = 32,
    ) -> Tensor:
        """Gradient-enabled encode for Phase 2 joint training."""
        return self._encoder.encode_trainable(
            texts,
            prompt_type,
            batch_size=batch_size,
        )

    def trainable_module(self) -> HfQueryEncoder:
        """Underlying HF module for joint optimizer param group."""
        return self._encoder.trainable_module()  # type: ignore[return-value]

    def supports_joint_training(self) -> bool:
        return True


class _SupportsJinaEncodeText(Protocol):
    def encode_text(
        self,
        texts: list[str],
        prompt_type: PromptType,
    ) -> npt.NDArray[np.float32]: ...


class _JinaFdeTextAdapter:
    """Thin adapter so frozen Jinavera satisfies TextEncoderProtocol."""

    def __init__(
        self,
        jina: _SupportsJinaEncodeText,
        dim: int,
        *,
        prefix_truncate: bool,
    ) -> None:
        self._jina = jina
        self._dim = dim
        self._prefix_truncate = prefix_truncate

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode_text(
        self,
        texts: str | Sequence[str],
        prompt_type: PromptType = PromptType.QUERY,
        *,
        batch_size: int = 8,
    ) -> npt.NDArray[np.float32]:
        del batch_size
        if isinstance(texts, str):
            texts = [texts]
        emb = self._jina.encode_text(list(texts), prompt_type)
        return _apply_prefix_truncate(
            emb if emb.ndim == 2 else emb.reshape(1, -1),
            self._dim,
            prefix_truncate=self._prefix_truncate,
        )


class _JinaLoraEncoderWrapper:
    """Expose JinaLoraQueryEncoder through QueryEncoderProtocol surface."""

    def __init__(self, encoder: JinaLoraQueryEncoder) -> None:
        self._encoder = encoder

    @property
    def name(self) -> str:
        return self._encoder.name

    @property
    def embedding_dim(self) -> int:
        return self._encoder.embedding_dim

    def encode_text(
        self,
        texts: str | Sequence[str],
        prompt_type: PromptType = PromptType.QUERY,
        *,
        batch_size: int = 8,
    ) -> npt.NDArray[np.float32]:
        return self._encoder.encode_text(
            texts,
            prompt_type,
            batch_size=batch_size,
        )

    def encode_passages(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
    ) -> npt.NDArray[np.float32]:
        return self.encode_text(texts, PromptType.PASSAGE, batch_size=batch_size)

    def encode_queries(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
    ) -> Tensor:
        return encode_text_as_tensor(
            self,
            texts,
            PromptType.QUERY,
            batch_size=batch_size,
        )

    def encode(self, texts: list[str]) -> Tensor:
        return self.encode_queries(texts)

    def encode_trainable(
        self,
        texts: list[str],
        prompt_type: PromptType = PromptType.QUERY,
        *,
        batch_size: int = 8,
    ) -> Tensor:
        return self._encoder.encode_trainable(
            texts,
            prompt_type,
            batch_size=batch_size,
        )

    def trainable_module(self) -> JinaLoraQueryEncoder:
        return self._encoder.trainable_module()  # type: ignore[return-value]

    def supports_joint_training(self) -> bool:
        return True

    def lora_state_dict(self) -> dict[str, Tensor]:
        return self._encoder.lora_state_dict()

    def load_lora_state_dict(self, state: dict[str, Tensor]) -> None:
        self._encoder.load_lora_state_dict(state)

    @property
    def lora_task(self) -> str:
        return self._encoder.lora_task

    @property
    def inner(self) -> JinaLoraQueryEncoder:
        return self._encoder
