from typing import Union

import torch
import transformers
from PIL import Image

from fde import base
from fde import config
from fde.hf.models import jina_embeddings_v4


_JINA_EMBEDDING_TASK = "retrieval"  # Must be one of ['retrieval', 'text-matching', 'code']


class Jinavera(base.MultiModalFdeEncoder):
    """
    FDE Encoder implementation using Jina Embeddings V4 model.
    """
    
    def _initialize_model(self, **kwargs) -> None:
        """Initialize Jina V4 model and tokenizer."""
        self.encoder_model = jina_embeddings_v4.JinaEmbeddingsV4Model.from_pretrained(
            self.encoder_model_path, 
            trust_remote_code=False
        ).to('cuda')
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.encoder_model_path, 
            trust_remote_code=False, 
            use_fast=True, 
            local_files_only=True
        )
    
    def _encode_texts_to_multivector(
        self, 
        texts: list[str], 
        prompt_type: config.PromptType
    ) -> list[torch.Tensor]:
        """Encode texts using Jina V4 model."""
        return self.encoder_model.encode_text(
            texts=texts,
            task=_JINA_EMBEDDING_TASK,
            return_multivector=True,
            prompt_name=prompt_type.value
        )
    
    def _encode_images_to_multivector(
        self, 
        images: list[Union[str, Image.Image]]
    ) -> list[torch.Tensor]:
        """Encode images using Jina V4 model."""
        return self.encoder_model.encode_image(
            images=images,
            task=_JINA_EMBEDDING_TASK,
            return_multivector=True
        )
    
    def __finalize__(self) -> None:
        """Clean up Jina model resources."""
        del self.encoder_model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
