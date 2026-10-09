"""Converts JAX functions directly into native onnx-light graphs."""

from .convert import to_onnx

__all__ = ["to_onnx"]
