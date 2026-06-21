from typing import Union

import muvfde
import numpy as np
import torch
import transformers
from PIL import Image

from fde import base
from fde import config
from fde.hf.models import jina_embeddings_v4


_JINA_EMBEDDING_TASK = (
    "retrieval"  # Must be one of ['retrieval', 'text-matching', 'code']
)


class Jinavera(base.MultiModalFdeEncoder):
    """FDE Encoder implementation using Jina Embeddings V4 model."""

    def _initialize_model(self, **kwargs) -> None:
        """Initialize Jina V4 model and tokenizer."""
        self.encoder_model = jina_embeddings_v4.JinaEmbeddingsV4Model.from_pretrained(
            self.encoder_model_path,
            trust_remote_code=False,
        ).to("cuda")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.encoder_model_path,
            trust_remote_code=False,
            use_fast=True,
            local_files_only=True,
        )

    def unload_model(self) -> None:
        """Release GPU weights to recover VRAM (reload via ensure_model_loaded)."""
        if getattr(self, "encoder_model", None) is not None:
            del self.encoder_model
            self.encoder_model = None
        if getattr(self, "tokenizer", None) is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_model_loaded(self) -> None:
        """Load Jina weights when unloaded after OOM recovery."""
        if getattr(self, "encoder_model", None) is None:
            self._initialize_model()

    def _encode_texts_to_multivector(
        self,
        texts: list[str],
        prompt_type: config.PromptType,
        *,
        encode_batch_size: int = 8,
    ) -> list[torch.Tensor]:
        """Encode texts using Jina V4 model."""
        self.ensure_model_loaded()
        return self.encoder_model.encode_text(
            texts=texts,
            task=_JINA_EMBEDDING_TASK,
            return_multivector=True,
            prompt_name=prompt_type.value,
            batch_size=encode_batch_size,
        )

    def encode_text(
        self,
        texts: Union[str, list[str]],
        prompt_type: config.PromptType = config.PromptType.QUERY,
        *,
        encode_batch_size: int = 8,
    ) -> np.ndarray:
        """Encode text(s) to FDE vectors with configurable Jina micro-batch size."""
        if isinstance(texts, str):
            texts = [texts]

        token_embs = self._encode_texts_to_multivector(
            texts,
            prompt_type,
            encode_batch_size=encode_batch_size,
        )
        fde_out = []
        for mat in token_embs:
            fde_embedding = muvfde.generate_fixed_dimensional_encoding(
                mat.to(dtype=torch.float32, device="cpu").numpy(),
                self.fde_configs[prompt_type],
            )
            fde_out.append(fde_embedding)

        return np.stack(fde_out, axis=0).astype(np.float32)

    def _encode_images_to_multivector(
        self,
        images: list[Union[str, Image.Image]],
    ) -> list[torch.Tensor]:
        """Encode images using Jina V4 model."""
        self.ensure_model_loaded()
        return self.encoder_model.encode_image(
            images=images,
            task=_JINA_EMBEDDING_TASK,
            return_multivector=True,
        )

    def __finalize__(self) -> None:
        """Clean up Jina model resources."""
        self.unload_model()
