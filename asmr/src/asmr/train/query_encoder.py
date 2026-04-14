"""HF-based query embeddings for MFARFieldAdapter / AggregationHead."""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class HfQueryEncoder(nn.Module):
    """Mean-pooled [CLS] or last-hidden-state embeddings (Contriever-style)."""

    def __init__(
        self,
        model_name: str = "facebook/contriever-msmarco",
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        super().__init__()
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

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> Tensor:
        """Return float32 tensor [B, H]."""
        out_vecs: list[Tensor] = []
        self._model.eval()
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            enc = self._tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            out = self._model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = F.normalize(pooled, p=2, dim=1)
            out_vecs.append(pooled.cpu())
        return torch.cat(out_vecs, dim=0).float()
