"""HuggingFace mean-pooled query encoder (Contriever-style)."""

from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from fde.config import PromptType


class HfQueryEncoder(nn.Module):
    """Mean-pooled last-hidden-state embeddings (Contriever-style)."""

    def __init__(
        self,
        model_name: str = "facebook/contriever-msmarco",
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._max_length = max_length
        dev = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._device = torch.device(dev)
        self._model.to(self._device)
        self._model.eval()

    @property
    def embedding_dim(self) -> int:
        hidden = getattr(self._model.config, "hidden_size", None)
        if hidden is None:
            raise ValueError("Model config missing hidden_size")
        return int(hidden)

    @property
    def name(self) -> str:
        slug = self._model_name.replace("/", "_")
        return f"contriever_{slug}"

    @property
    def device(self) -> torch.device:
        return self._device

    def _encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        grad: bool,
    ) -> Tensor:
        out_vecs: list[Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            enc = self._tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            enc = {key: val.to(self._device) for key, val in enc.items()}
            if grad:
                out = self._model(**enc)
            else:
                with torch.no_grad():
                    out = self._model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = F.normalize(pooled, p=2, dim=1)
            out_vecs.append(pooled.float())
        return torch.cat(out_vecs, dim=0)

    def encode_text(
        self,
        texts: str | Sequence[str],
        prompt_type: PromptType = PromptType.QUERY,
        *,
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        """Return float32 numpy [B, H] on CPU (inference-only)."""
        del prompt_type  # Contriever bi-encoder: query/passage pooling is identical.
        if isinstance(texts, str):
            texts = [texts]
        self._model.eval()
        vecs = self._encode_batch(texts, batch_size=batch_size, grad=False)
        return vecs.cpu().numpy().astype(np.float32)

    def encode_trainable(
        self,
        texts: Sequence[str],
        prompt_type: PromptType = PromptType.QUERY,
        *,
        batch_size: int = 32,
    ) -> Tensor:
        """Return float32 tensor [B, H] on device with gradients enabled."""
        del prompt_type
        self._model.train()
        return self._encode_batch(texts, batch_size=batch_size, grad=True)

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> Tensor:
        """Deprecated: prefer encode_text(..., PromptType.QUERY)."""
        return torch.from_numpy(
            self.encode_text(texts, PromptType.QUERY, batch_size=batch_size),
        )

    def trainable_module(self) -> nn.Module:
        """Underlying HF module for joint optimizer param group."""
        return self._model

    def supports_joint_training(self) -> bool:
        return True
