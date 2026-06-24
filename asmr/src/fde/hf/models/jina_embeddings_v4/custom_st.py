from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import requests
import torch
from PIL import Image
from torch import nn
from transformers import AutoConfig, AutoModel, AutoProcessor


class Transformer(nn.Module):
    save_in_root: bool = True

    def __init__(
        self,
        model_name_or_path: str = "jinaai/jina-embeddings-v4",
        max_seq_length: Optional[int] = None,
        config_args: Optional[Dict[str, Any]] = None,
        model_args: Optional[Dict[str, Any]] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        **kwargs,
    ) -> None:
        super(Transformer, self).__init__()
        if backend != "torch":
            raise ValueError(
                f"Backend '{backend}' is not supported, please use 'torch' instead"
            )
        config_kwargs = config_args or {}
        model_kwargs = model_args or {}
        tokenizer_kwargs = tokenizer_args or {}

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, **config_kwargs
        )
        self.default_task = model_args.pop("default_task", None)
        if self.default_task and self.default_task not in self.config.task_names:
            raise ValueError(
                f"Invalid task: {self.default_task}. Must be one of {self.config.task_names}."
            )

        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=self.config, cache_dir=cache_dir, **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            use_fast=True,
            **tokenizer_kwargs,
        )
        self.max_seq_length = max_seq_length or 8192

    def tokenize(
        self, texts: List[Union[str, Image.Image]], padding: Union[str, bool] = True
    ) -> Dict[str, torch.Tensor]:
        encoding = {}
        text_indices = []
        image_indices = []
        for i, text in enumerate(texts):
            if isinstance(text, str):
                # Remove Query: or Passage: prefixes when checking for URLs or file paths
                clean_text = text
                if text.startswith("Query: "):
                    clean_text = text[len("Query: ") :]
                elif text.startswith("Passage: "):
                    clean_text = text[len("Passage: ") :]

                if clean_text.startswith("http"):
                    response = requests.get(clean_text)
                    texts[i] = Image.open(BytesIO(response.content)).convert("RGB")
                    image_indices.append(i)
                else:
                    try:
                        if Path(clean_text).is_file():
                            texts[i] = Image.open(clean_text).convert("RGB")
                            image_indices.append(i)
                        else:
                            text_indices.append(i)
                    except Exception:
                        text_indices.append(i)
            elif isinstance(text, Image.Image):
                image_indices.append(i)
            else:
                raise ValueError(f"Invalid input type: {type(text)}")
        if text_indices:
            _texts = [texts[i] for i in text_indices]
            text_features = self.processor.process_texts(
                _texts, max_length=self.max_seq_length
            )
            for key, value in text_features.items():
                encoding[f"text_{key}"] = value
            encoding["text_indices"] = text_indices

        if image_indices:
            _images = [texts[i] for i in image_indices]
            img_features = self.processor.process_images(_images)
            for key, value in img_features.items():
                encoding[f"image_{key}"] = value
            encoding["image_indices"] = image_indices

        return encoding

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        task: Optional[str] = None,
        truncate_dim: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()

        if task is None:
            if self.default_task is None:
                raise ValueError(
                    "Task must be specified before encoding data. You can set it either during "
                    "loading the model (e.g., model_kwargs={'default_task': 'retrieval'}) or "
                    "pass it as an argument to the encode method (e.g., model.encode(texts, task='retrieval'))."
                )
            task = self.default_task
        else:
            if task not in self.config.task_names:
                raise ValueError(
                    f"Invalid task: {task}. Must be one of {self.config.task_names}."
                )

        device = self.model.device.type
        all_embeddings = []

        with torch.no_grad():
            if any(k.startswith("text_") for k in features.keys()):
                text_batch = {
                    k[len("text_") :]: v.to(device)
                    for k, v in features.items()
                    if k.startswith("text_") and k != "text_indices"
                }
                text_indices = features.get("text_indices", [])
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    text_embeddings = self.model(
                        **text_batch, task_label=task
                    ).single_vec_emb
                    if truncate_dim:
                        text_embeddings = text_embeddings[:, :truncate_dim]
                        text_embeddings = torch.nn.functional.normalize(
                            text_embeddings, p=2, dim=-1
                        )
                for i, embedding in enumerate(text_embeddings):
                    all_embeddings.append((text_indices[i], embedding))

            if any(k.startswith("image_") for k in features.keys()):
                image_batch = {
                    k[len("image_") :]: v.to(device)
                    for k, v in features.items()
                    if k.startswith("image_") and k != "image_indices"
                }
                image_indices = features.get("image_indices", [])

                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    img_embeddings = self.model(
                        **image_batch, task_label=task
                    ).single_vec_emb
                    if truncate_dim:
                        img_embeddings = img_embeddings[:, :truncate_dim]
                        img_embeddings = torch.nn.functional.normalize(
                            img_embeddings, p=2, dim=-1
                        )

                for i, embedding in enumerate(img_embeddings):
                    all_embeddings.append((image_indices[i], embedding))

        if not all_embeddings:
            raise RuntimeError("No embeddings were generated")

        all_embeddings.sort(key=lambda x: x[0])  # sort by original index
        combined_embeddings = torch.stack([emb for _, emb in all_embeddings])
        features["sentence_embedding"] = combined_embeddings

        return features

    @classmethod
    def load(cls, input_path: str) -> "Transformer":
        return cls(model_name_or_path=input_path)
