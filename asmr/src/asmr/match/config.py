"""Configuration dataclass for the image ranking module."""

from dataclasses import dataclass


@dataclass
class RankerConfig:
    """Configuration for ImageRanker.

    Attributes:
        field_name: Field name used for the image index inside QueryRouter.
        top_k: Default number of top results returned by rank().
        image_load_timeout_s: HTTP timeout (seconds) when loading images by URL.
        device: Torch device string passed to the aggregation head ('cpu' or 'cuda').
    """

    field_name: str = "image"
    top_k: int = 5
    image_load_timeout_s: float = 10.0
    device: str = "cpu"
