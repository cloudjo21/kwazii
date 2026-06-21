"""STaRK-Amazon dataset schema — auto-registers on import."""

from asmr.datasets import register_dataset
from asmr.datasets.stark_amazon.schema import STARK_AMAZON_SCHEMA, StarkAmazonSchema

register_dataset(STARK_AMAZON_SCHEMA)

__all__ = ["STARK_AMAZON_SCHEMA", "StarkAmazonSchema"]
