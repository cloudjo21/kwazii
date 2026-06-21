import abc
import numpy as np

import muvfde
import torch
from typing import List, Optional, Union
from PIL import Image

from fde import config
from fde.config import PromptType


class BaseFdeEncoder(torch.nn.Module, abc.ABC):
    """
    Base class for Fixed Dimensional Encoding (FDE) encoders that wrap
    multi-modal conditional generation models.

    This class provides a common interface for encoding text and images
    into fixed-dimensional representations using various underlying models.
    """

    def __init__(self, encoder_model_path: str, **kwargs):
        super().__init__()
        self.encoder_model_path = encoder_model_path
        fde_output_dim = kwargs.pop("fde_output_dim", None)
        self.fde_output_dim = fde_output_dim
        if fde_output_dim is not None:
            self.fde_configs = {
                PromptType.QUERY: config.FdeConfig.apply_with_fde_output_dim(
                    PromptType.QUERY,
                    fde_output_dim,
                ),
                PromptType.PASSAGE: config.FdeConfig.apply_with_fde_output_dim(
                    PromptType.PASSAGE,
                    fde_output_dim,
                ),
            }
        else:
            self.fde_configs = {
                PromptType.QUERY: config.FdeConfig.apply_with_prompt_type(
                    PromptType.QUERY
                ),
                PromptType.PASSAGE: config.FdeConfig.apply_with_prompt_type(
                    PromptType.PASSAGE
                ),
            }
        self._initialize_model(**kwargs)

    @abc.abstractmethod
    def _initialize_model(self, **kwargs) -> None:
        """Initialize the underlying encoder model and tokenizer/processor."""
        pass

    @abc.abstractmethod
    def _encode_texts_to_multivector(
        self, texts: List[str], prompt_type: PromptType
    ) -> List[torch.Tensor]:
        """
        Encode texts to multi-vector embeddings using the underlying model.

        Returns:
            List of tensors, one per input text with shape [seq_len, hidden_dim]
        """
        pass

    @abc.abstractmethod
    def _encode_images_to_multivector(
        self, images: List[Union[str, Image.Image]]
    ) -> List[torch.Tensor]:
        """
        Encode images to multi-vector embeddings using the underlying model.

        Returns:
            List of tensors, one per input image with shape [seq_len, hidden_dim]
        """
        pass

    def encode_text(
        self, texts: Union[str, List[str]], prompt_type: PromptType = PromptType.QUERY
    ) -> np.ndarray:
        """
        Encode text(s) into fixed-dimensional representations.

        Args:
            texts: Single text string or list of text strings
            prompt_type: Type of prompt (QUERY or PASSAGE)

        Returns:
            Fixed-dimensional embeddings as numpy array
        """
        if isinstance(texts, str):
            texts = [texts]

        # Get multi-vector embeddings from the underlying model
        token_embs = self._encode_texts_to_multivector(texts, prompt_type)

        # Apply FDE transformation
        fde_out = []
        for mat in token_embs:
            fde_embedding = muvfde.generate_fixed_dimensional_encoding(
                mat.to(dtype=torch.float32, device="cpu").numpy(),
                self.fde_configs[prompt_type],
            )
            fde_out.append(fde_embedding)

        return np.stack(fde_out, axis=0).astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """FDE output dimension for the default QUERY prompt type."""
        if self.fde_output_dim is not None:
            return int(self.fde_output_dim)
        cfg = self.fde_configs[PromptType.QUERY]
        dim = getattr(cfg, "dimension", None)
        if dim is not None:
            return int(dim)
        probe = self.encode_text(["probe"], PromptType.QUERY)
        return int(probe.shape[1])

    def encode_image(
        self, images: Union[str, Image.Image, List[Union[str, Image.Image]]]
    ) -> np.ndarray:
        """
        Encode image(s) into fixed-dimensional representations.

        Args:
            images: Single image or list of images (PIL Images, URLs, or file paths)

        Returns:
            Fixed-dimensional embeddings as numpy array
        """
        if not isinstance(images, list):
            images = [images]

        # Get multi-vector embeddings from the underlying model
        token_embs = self._encode_images_to_multivector(images)

        # Apply FDE transformation (using PASSAGE config for images)
        fde_out = []
        for mat in token_embs:
            fde_embedding = muvfde.generate_fixed_dimensional_encoding(
                mat.to(dtype=torch.float32, device="cpu").numpy(),
                self.fde_configs[PromptType.PASSAGE],
            )
            fde_out.append(fde_embedding)

        return np.stack(fde_out, axis=0)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> torch.Tensor:
        """Calculate similarity between two sets of embeddings."""
        return torch.from_numpy(a @ b.T)

    @abc.abstractmethod
    def __finalize__(self) -> None:
        """Clean up resources and clear GPU memory."""
        pass


class MultiModalFdeEncoder(BaseFdeEncoder):
    """
    Extended base class for multi-modal FDE encoders that support both text and image encoding.
    """

    def encode(
        self,
        inputs: Union[
            str,
            List[str],
            Image.Image,
            List[Image.Image],
            List[Union[str, Image.Image]],
        ],
        prompt_type: PromptType = PromptType.QUERY,
        input_type: Optional[str] = None,
    ) -> np.ndarray:
        """
        Unified encoding method that handles both text and image inputs.

        Args:
            inputs: Text strings, images, or mixed list
            prompt_type: Type of prompt for text inputs
            input_type: Force input type ('text' or 'image'), auto-detect if None

        Returns:
            Fixed-dimensional embeddings as numpy array
        """
        # Auto-detect input type if not specified
        if input_type is None:
            if isinstance(inputs, str):
                input_type = "text"
            elif isinstance(inputs, Image.Image):
                input_type = "image"
            elif isinstance(inputs, list) and len(inputs) > 0:
                first_item = inputs[0]
                if isinstance(first_item, str):
                    input_type = "text"
                elif isinstance(first_item, Image.Image):
                    input_type = "image"
                else:
                    raise ValueError(
                        "Mixed input types not supported in unified encode method"
                    )
            else:
                raise ValueError("Cannot auto-detect input type")

        if input_type == "text":
            return self.encode_text(inputs, prompt_type)
        elif input_type == "image":
            return self.encode_image(inputs)
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
