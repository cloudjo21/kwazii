"""Unified text encoder protocol layer for asmr."""

from asmr.encode.protocol import (
    LoraTrainableTextEncoderProtocol,
    MultiModalEncoderProtocol,
    NamedTextEncoderProtocol,
    QueryEncoderProtocol,
    TextEncoderProtocol,
    TrainableTextEncoderProtocol,
    encode_text_as_tensor,
)

__all__ = [
    "LoraTrainableTextEncoderProtocol",
    "MultiModalEncoderProtocol",
    "NamedTextEncoderProtocol",
    "QueryEncoderProtocol",
    "TextEncoderProtocol",
    "TrainableTextEncoderProtocol",
    "encode_text_as_tensor",
]
