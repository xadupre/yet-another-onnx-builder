"""
Internal einsum decomposition utilities.

This package is a self-contained port of the einsum decomposition logic
originally developed in:
https://github.com/sdpython/onnx-extended/tree/main/onnx_extended/tools/einsum
(MIT licence).

Public entry points (used by :func:`yobx.helpers.einsum_helper.decompose_einsum`):

* :func:`decompose_einsum_equation` — decomposes an equation into a
  :class:`GraphEinsumSubOp` graph.
* :class:`GraphEinsumSubOp` — a graph of :class:`EinsumSubOp` nodes that can
  evaluate itself or be converted to an :class:`onnx.ModelProto`.
* :class:`EinsumSubOp` — a single node in the decomposition graph.
"""

from .einsum_impl import decompose_einsum_equation, analyse_einsum_equation
from .einsum_impl_classes import EinsumSubOp, GraphEinsumSubOp
from .einsum_impl_ext import (
    numpy_extended_dot,
    numpy_extended_dot_python,
    numpy_extended_dot_matrix,
    numpy_diagonal,
)

__all__ = [
    "EinsumSubOp",
    "GraphEinsumSubOp",
    "analyse_einsum_equation",
    "decompose_einsum_equation",
    "numpy_diagonal",
    "numpy_extended_dot",
    "numpy_extended_dot_matrix",
    "numpy_extended_dot_python",
]
