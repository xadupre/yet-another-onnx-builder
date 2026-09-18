"""Defines the inference modes accepted by the native shape adapter."""

from enum import IntEnum


class InferenceMode(IntEnum):
    """Selects no inference, native shape/type inference, or cost estimation."""

    NOTHING = 0
    SHAPE = 1
    TYPE = 2
    COST = 16
