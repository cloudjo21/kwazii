"""STaRK-Prime dataset schema — auto-registers on import."""

from asmr.datasets import register_dataset
from asmr.datasets.stark_prime.schema import STARK_PRIME_SCHEMA, StarkPrimeSchema

register_dataset(STARK_PRIME_SCHEMA)

__all__ = ["STARK_PRIME_SCHEMA", "StarkPrimeSchema"]
