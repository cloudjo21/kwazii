from dataclasses import dataclass
from typing import Union, Optional, List, Dict, TYPE_CHECKING
from PIL import Image

if TYPE_CHECKING:
    from asmr.index.config import FieldConfig


@dataclass
class QueryContent:
    """Content wrapper with data type information"""

    content: Union[str, Image.Image]
    data_type: str  # 'text' or 'image'

    def __init__(
        self, content: Union[str, Image.Image], data_type: Optional[str] = None
    ):
        self.content = content
        if data_type is None:
            # Auto-detect data type
            if isinstance(content, str):
                self.data_type = "text"
            elif isinstance(content, Image.Image):
                self.data_type = "image"
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        else:
            self.data_type = data_type


@dataclass
class Query:
    """Query class to support text and image queries simultaneously"""

    text: Optional[QueryContent] = None
    image: Optional[QueryContent] = None

    def __post_init__(self):
        if self.text is None and self.image is None:
            raise ValueError(
                "Query must have at least one content type (text or image)"
            )

    def has_text(self) -> bool:
        """Check if this query has text content"""
        return self.text is not None

    def has_image(self) -> bool:
        """Check if this query has image content"""
        return self.image is not None

    def is_multimodal(self) -> bool:
        """Check if this query has both text and image content"""
        return self.has_text() and self.has_image()

    def get_text(self) -> str:
        """Get text content, raises error if no text"""
        if self.text is None:
            raise ValueError("This query has no text content")
        return self.text.content

    def get_image(self) -> Image.Image:
        """Get image content, raises error if no image"""
        if self.image is None:
            raise ValueError("This query has no image content")
        return self.image.content

    def get_text_data_type(self) -> str:
        """Get text data type"""
        if self.text is None:
            raise ValueError("This query has no text content")
        return self.text.data_type

    def get_image_data_type(self) -> str:
        """Get image data type"""
        if self.image is None:
            raise ValueError("This query has no image content")
        return self.image.data_type

    def get_target_fields(
        self, field_configs: Dict[str, "FieldConfig"]
    ) -> List["FieldConfig"]:
        """Get target field configurations based on query content types"""
        from asmr.index.config import RepresentationType

        target_fields = []

        # For text content, find compatible text fields
        if self.has_text():
            for field_name, config in field_configs.items():
                # Text can match with both sparse and dense text fields
                if hasattr(config, "supports_text") or "text" in field_name.lower():
                    target_fields.append(config)

        # For image content, find compatible image fields
        if self.has_image():
            for field_name, config in field_configs.items():
                # Images typically use dense representation
                if config.representation_type == RepresentationType.DENSE and (
                    hasattr(config, "supports_image") or "image" in field_name.lower()
                ):
                    target_fields.append(config)

        return target_fields

    @classmethod
    def from_text(cls, text: str, data_type: str = "text") -> "Query":
        """Create a text-only query"""
        return cls(text=QueryContent(text, data_type))

    @classmethod
    def from_image(
        cls, image: Union[str, Image.Image], data_type: str = "image"
    ) -> "Query":
        """Create an image-only query"""
        if isinstance(image, str):
            # If it's a string, assume it's a file path and load the image
            image = Image.open(image)
        return cls(image=QueryContent(image, data_type))

    @classmethod
    def from_multimodal(
        cls,
        text: str,
        image: Union[str, Image.Image],
        text_data_type: str = "text",
        image_data_type: str = "image",
    ) -> "Query":
        """Create a multimodal query with both text and image"""
        if isinstance(image, str):
            image = Image.open(image)
        return cls(
            text=QueryContent(text, text_data_type),
            image=QueryContent(image, image_data_type),
        )
