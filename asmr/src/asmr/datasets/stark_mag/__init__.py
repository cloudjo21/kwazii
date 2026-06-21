"""STaRK-MAG dataset schema — auto-registers on import."""

from asmr.datasets import register_dataset
from asmr.datasets.stark_mag.schema import STARK_MAG_SCHEMA, StarkMagSchema

register_dataset(STARK_MAG_SCHEMA)

__all__ = ["STARK_MAG_SCHEMA", "StarkMagSchema"]
