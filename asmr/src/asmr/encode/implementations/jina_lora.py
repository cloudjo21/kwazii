"""Jina V4 LoRA-only trainable query encoder for MFAR Phase 2."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor, nn

from fde.config import PromptType


@dataclass(frozen=True)
class JinaLoraConfig:
    """Configuration for Jina V4 LoRA-only query encoder training."""

    model_path: str = "jinaai/jina-embeddings-v4"
    task: str = "retrieval"
    truncate_dim: int = 1024
    lora_lr: float = 1e-4
    train_task_adapter: str = "retrieval"
    adapter_name: str = ""
    inference_use_fde: bool = False


class JinaLoraQueryEncoder(nn.Module):
    """Jina V4 query encoder with LoRA-only trainable params for MFAR Phase 2."""

    def __init__(
        self,
        cfg: JinaLoraConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        super().__init__()
        from fde.hf.models import jina_embeddings_v4

        self._cfg = cfg or JinaLoraConfig()
        dev = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._device = torch.device(dev)
        self._model = jina_embeddings_v4.JinaEmbeddingsV4Model.from_pretrained(
            self._cfg.model_path,
            trust_remote_code=False,
        )
        self._model.to(self._device)
        self._active_adapter_name = self._select_pretrained_adapter()
        self.freeze_non_lora()
        self._jinavera = None

    def _select_pretrained_adapter(self) -> str:
        """Use existing Jina PEFT adapter (Strategy A: in-place LoRA FT)."""
        cfg = self._cfg
        peft_names = list(self._model.peft_config.keys())
        if cfg.adapter_name and cfg.adapter_name in peft_names:
            chosen = cfg.adapter_name
        elif peft_names:
            chosen = peft_names[0]
        else:
            msg = "Jina PeftModel has no loaded adapters"
            raise ValueError(msg)
        self._model.set_adapter(chosen)
        return chosen

    @property
    def embedding_dim(self) -> int:
        return self._cfg.truncate_dim

    @property
    def name(self) -> str:
        return f"jinavera_lora_{self._cfg.truncate_dim}_{self._active_adapter_name}"

    @property
    def lora_task(self) -> str:
        return self._cfg.train_task_adapter

    @property
    def device(self) -> torch.device:
        return self._device

    def _normalize_vectors(self, vecs: Tensor) -> Tensor:
        if vecs.shape[-1] > self._cfg.truncate_dim:
            vecs = vecs[:, : self._cfg.truncate_dim]
        return F.normalize(vecs.float(), p=2, dim=-1)

    def _encode_single_vector_batch(
        self,
        texts: Sequence[str],
        prompt_type: PromptType,
        *,
        batch_size: int,
        grad: bool,
    ) -> Tensor:
        from fde.hf.models.jina_embeddings_v4.modeling_jina_embeddings_v4 import (
            PREFIX_DICT,
        )

        cfg = self._cfg
        self._model.task = cfg.task
        prefix = (
            PREFIX_DICT[prompt_type.value]
            if cfg.task != "text-matching"
            else PREFIX_DICT["query"]
        )
        out_vecs: list[Tensor] = []
        text_list = list(texts)
        processor = self._model.processor
        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]
            batch_feat = processor.process_texts(batch, prefix=prefix)
            batch_feat = {key: val.to(self._device) for key, val in batch_feat.items()}
            if grad:
                self._model.train()
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.bfloat16,
                ):
                    output = self._model(**batch_feat, task_label=cfg.task)
                emb = output.single_vec_emb.float()
            else:
                self._model.eval()
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=self._device.type,
                        dtype=torch.bfloat16,
                    ),
                ):
                    output = self._model(**batch_feat, task_label=cfg.task)
                emb = output.single_vec_emb.float()
            out_vecs.append(self._normalize_vectors(emb))
        return torch.cat(out_vecs, dim=0)

    def encode_text(
        self,
        texts: str | Sequence[str],
        prompt_type: PromptType = PromptType.QUERY,
        *,
        batch_size: int = 8,
    ) -> npt.NDArray[np.float32]:
        """Inference-only encode; single-vector path by default."""
        if isinstance(texts, str):
            texts = [texts]
        if self._cfg.inference_use_fde:
            return self._encode_text_fde(texts, prompt_type)
        self._model.eval()
        vecs = self._encode_single_vector_batch(
            texts,
            prompt_type,
            batch_size=batch_size,
            grad=False,
        )
        return vecs.detach().cpu().numpy().astype(np.float32)

    def _encode_text_fde(
        self,
        texts: Sequence[str],
        prompt_type: PromptType,
    ) -> npt.NDArray[np.float32]:
        from fde.models.jinavera import Jinavera

        if self._jinavera is None:
            self._jinavera = Jinavera(encoder_model_path=self._cfg.model_path)
        return self._jinavera.encode_text(list(texts), prompt_type).astype(np.float32)

    def encode_trainable(
        self,
        texts: Sequence[str],
        prompt_type: PromptType = PromptType.QUERY,
        *,
        batch_size: int = 8,
    ) -> Tensor:
        """Single-vector Jina forward with grad; LoRA params only in optimizer."""
        self._model.train()
        return self._encode_single_vector_batch(
            texts,
            prompt_type,
            batch_size=batch_size,
            grad=True,
        )

    def trainable_module(self) -> nn.Module:
        return self._model

    def supports_joint_training(self) -> bool:
        return True

    def freeze_non_lora(self) -> None:
        """Freeze base weights; enable grad only on active LoRA matrices."""
        cfg = self._cfg
        adapter_tag = self._active_adapter_name
        for name, param in self._model.named_parameters():
            trainable = (
                "lora_" in name
                and adapter_tag in name
                and cfg.train_task_adapter in name
            )
            param.requires_grad = trainable

    def lora_parameters(self) -> list[Tensor]:
        """Trainable LoRA parameters for optimizer grouping."""
        return [param for param in self._model.parameters() if param.requires_grad]

    def lora_state_dict(self) -> dict[str, Tensor]:
        """Export trainable LoRA weights only."""
        return get_peft_model_state_dict(
            self._model,
            adapter_name=self._active_adapter_name,
        )

    def load_lora_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load LoRA weights for eval merge."""
        set_peft_model_state_dict(
            self._model,
            state,
            adapter_name=self._active_adapter_name,
        )
        self._model.set_adapter(self._active_adapter_name)
