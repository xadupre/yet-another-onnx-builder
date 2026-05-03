"""
Internal einsum decomposition utilities.

Public entry points (used by :func:`yobx.helpers.einsum_helper.decompose_einsum`):

* :func:`decompose_einsum_equation` — decomposes an equation into a
  :class:`GraphEinsumSubOp` graph.
* :class:`GraphEinsumSubOp` — a graph of :class:`EinsumSubOp` nodes that can
  evaluate itself or be converted to an :class:`onnx.ModelProto`.
* :class:`EinsumSubOp` — a single node in the decomposition graph.
* :func:`decompose_einsum_2inputs` — a completely independent algorithm that
  builds an ONNX graph directly from a 2-operand einsum equation.
* :class:`EinsumBuilder` — the stateful ONNX node/initializer accumulator used
  by :func:`decompose_einsum_2inputs`.
"""

from .einsum_impl import decompose_einsum_equation, analyse_einsum_equation
from .einsum_impl_classes import EinsumSubOp, GraphEinsumSubOp
from .einsum_impl_ext import (
    numpy_extended_dot,
    numpy_extended_dot_python,
    numpy_extended_dot_matrix,
    numpy_diagonal,
)
from .einsum_2_onnx import (
    const_int64,
    decompose_einsum_2inputs,
    EinsumBuilder,
    is_identity_perm,
    parse_2input_equation,
)

__all__ = [
    "EinsumBuilder",
    "EinsumSubOp",
    "GraphEinsumSubOp",
    "analyse_einsum_equation",
    "const_int64",
    "decompose_einsum_2inputs",
    "decompose_einsum_equation",
    "is_identity_perm",
    "numpy_diagonal",
    "numpy_extended_dot",
    "numpy_extended_dot_matrix",
    "numpy_extended_dot_python",
    "parse_2input_equation",
]
