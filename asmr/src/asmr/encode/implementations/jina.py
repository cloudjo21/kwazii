"""Frozen Jinavera query encoder for inference, indexing, and evaluation."""

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from fde.config import PromptType

from asmr.encode.adapters import (
    _JinaFdeTextAdapter,
    _apply_prefix_truncate,
    _resolve_jina_dims,
)
from asmr.encode.implementations.jina_lora import JinaLoraConfig, JinaLoraQueryEncoder
from asmr.encode.jina_gpu_oom import (
    JINA_DEFAULT_ENCODE_BATCH,
    JINA_OOM_ENCODE_BATCH,
    clear_cuda_cache,
    is_cuda_oom,
)
from asmr.encode.protocol import encode_text_as_tensor

logger = logging.getLogger(__name__)


class JinaQueryEncoder:
    """Frozen Jinavera for inference, index, and eval (default query encoder)."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        truncate_dim: int = 1024,
        fde_output_dim: int | None = 1024,
        lora_checkpoint: Path | str | None = None,
    ) -> None:
        from fde.models.jinavera import Jinavera

        import os

        _default_model = os.getenv("ENCODER_MODEL_PATH", "jinaai/jina-embeddings-v4")

        mrl_dim, resolved_fde = _resolve_jina_dims(truncate_dim, fde_output_dim)
        self._model_path = model_path or _default_model
        self._prompt_query = PromptType.QUERY
        self._prompt_passage = PromptType.PASSAGE
        self._truncate_dim = mrl_dim
        self._fde_output_dim = resolved_fde
        jina_kwargs: dict[str, object] = {}
        if resolved_fde is not None:
            jina_kwargs["fde_output_dim"] = resolved_fde
        self._jina = Jinavera(encoder_model_path=self._model_path, **jina_kwargs)
        native_dim = int(self._jina.embedding_dim)
        if resolved_fde is not None:
            self._dim = native_dim
            self._prefix_truncate = False
        else:
            self._dim = min(mrl_dim, native_dim) if mrl_dim > 0 else native_dim
            self._prefix_truncate = self._dim < native_dim
        self._lora_encoder: JinaLoraQueryEncoder | None = None
        self._jina_encode_batch_size = JINA_DEFAULT_ENCODE_BATCH
        self._jina_fieldwise_gpu = False
        if lora_checkpoint is not None:
            self._load_lora(Path(lora_checkpoint))

    def release_gpu(self) -> None:
        """Unload Jinavera weights from GPU (index-build OOM recovery)."""
        self._jina.unload_model()

    def ensure_gpu(self) -> None:
        """Reload Jinavera on GPU after release_gpu."""
        self._jina.ensure_model_loaded()

    @property
    def jina_fieldwise_gpu(self) -> bool:
        """When True, index build releases GPU between fields."""
        return self._jina_fieldwise_gpu

    def set_jina_encode_batch_size(self, batch_size: int) -> None:
        """Set Jina micro-batch size for FDE multivector encoding."""
        self._jina_encode_batch_size = batch_size

    def enable_fieldwise_gpu(self) -> None:
        """Release Jina GPU between index fields after OOM recovery."""
        self._jina_fieldwise_gpu = True

    def _encode_jina_batch(
        self,
        batch: list[str],
        prompt_type: PromptType,
    ) -> npt.NDArray[np.float32]:
        """Encode one batch with CUDA OOM fallback (batch 4, then reload)."""
        encode_bs = self._jina_encode_batch_size
        try:
            return self._jina.encode_text(
                batch,
                prompt_type,
                encode_batch_size=encode_bs,
            )
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            clear_cuda_cache()
            logger.warning(
                "Jina CUDA OOM at encode_batch_size=%d; retry with %d",
                encode_bs,
                JINA_OOM_ENCODE_BATCH,
            )
            self._jina_encode_batch_size = JINA_OOM_ENCODE_BATCH
            try:
                return self._jina.encode_text(
                    batch,
                    prompt_type,
                    encode_batch_size=JINA_OOM_ENCODE_BATCH,
                )
            except Exception as exc2:
                if not is_cuda_oom(exc2):
                    raise
            clear_cuda_cache()
            logger.warning(
                "Jina CUDA OOM persists; unload/reload model (batch=%d)",
                JINA_OOM_ENCODE_BATCH,
            )
            self.release_gpu()
            self.ensure_gpu()
            self.enable_fieldwise_gpu()
            return self._jina.encode_text(
                batch,
                prompt_type,
                encode_batch_size=JINA_OOM_ENCODE_BATCH,
            )

    @property
    def name(self) -> str:
        suffix = "_lora" if self._lora_encoder is not None else ""
        return f"jinavera_{self._dim}{suffix}"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def fde_output_dim(self) -> int | None:
        return self._fde_output_dim

    def _finalize_vectors(
        self, vectors: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Apply legacy prefix truncate only when FDE native dim is not configured."""
        return _apply_prefix_truncate(
            vectors,
            self._dim,
            prefix_truncate=self._prefix_truncate,
        )

    def _active_encoder(self) -> "_JinaFdeTextAdapter | JinaLoraQueryEncoder":
        if self._lora_encoder is not None:
            return self._lora_encoder
        return _JinaFdeTextAdapter(
            self._jina,
            self._dim,
            prefix_truncate=self._prefix_truncate,
        )

    def encode_text(
        self,
        texts: str | Sequence[str],
        prompt_type: PromptType = PromptType.QUERY,
        *,
        batch_size: int = 8,
    ) -> npt.NDArray[np.float32]:
        if self._lora_encoder is not None:
            return self._lora_encoder.encode_text(
                texts,
                prompt_type,
                batch_size=batch_size,
            )
        if isinstance(texts, str):
            texts = [texts]
        chunks: list[npt.NDArray[np.float32]] = []
        text_list = list(texts)
        micro_batch = min(batch_size, self._jina_encode_batch_size)
        for start in range(0, len(text_list), micro_batch):
            batch = text_list[start : start + micro_batch]
            emb = self._encode_jina_batch(batch, prompt_type)
            chunks.append(self._finalize_vectors(emb))
        if not chunks:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

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
        """Legacy SupportsQueryEncode compatibility."""
        return self.encode_queries(texts)

    def supports_joint_training(self) -> bool:
        return False

    def with_lora(self, checkpoint: Path | str) -> "JinaQueryEncoder":
        """Return a copy with LoRA adapter loaded for post-train eval."""
        copy = JinaQueryEncoder(
            model_path=self._model_path,
            truncate_dim=self._truncate_dim,
            fde_output_dim=self._fde_output_dim,
        )
        copy._load_lora(Path(checkpoint))
        return copy

    def _load_lora(self, checkpoint: Path) -> None:
        cfg = JinaLoraConfig(model_path=self._model_path, truncate_dim=self._dim)
        self._lora_encoder = JinaLoraQueryEncoder(cfg)
        state = torch.load(checkpoint, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._lora_encoder.load_lora_state_dict(state)
