"""Registers the built-in JAX primitive lowerings."""

from . import _control_flow, _elementwise, _linalg, _tensor
from ._registry import DynamicSources, lower_jaxpr

__all__ = ["DynamicSources", "lower_jaxpr"]
