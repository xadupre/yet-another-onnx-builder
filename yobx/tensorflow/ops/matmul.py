"""
Converter for the TF ``MatMul`` / ``BatchMatMulV2`` / ``BatchMatMul`` op →
ONNX ``MatMul``.

Matrix multiplication
---------------------
``MatMul``, ``BatchMatMulV2``, ``BatchMatMul``

All three TF op variants produce standard matrix multiplication (or batched
matrix multiplication) and map directly to a single ONNX ``MatMul`` node.

Transposition / adjoint
-----------------------
TF supports two families of "reverse-input" attributes that indicate the
operands should be transposed before multiplying:

* ``transpose_a`` / ``transpose_b`` — used by 2-D ``MatMul``
* ``adj_x`` / ``adj_y`` — used by ``BatchMatMulV2`` and ``BatchMatMul``

For real-valued tensors the *adjoint* is identical to the *transpose*, so
both families are handled identically: when an attribute is ``True`` an ONNX
``Transpose`` node that swaps the **last two dimensions** is inserted before
the ``MatMul``.

Because the batch dimensions can be of arbitrary rank the permutation is
computed dynamically by :func:`_transpose_last_two` rather than using
hard-coded indices like ``[-1, -2]``.

Example
-------
The following TF graph fragment:

.. code-block:: python

    # batched matmul with transposed B: result[b, i, j] = sum_k A[b, i, k] * B[b, j, k]
    tf.matmul(A, B, adjoint_b=True)   # BatchMatMulV2 with adj_y=True

is converted to:

.. code-block:: text

    Transpose(B, perm=[0, 2, 1])   # swap last two dims
    MatMul(A, transposed_B)
"""

from typing import Any, Dict, List
import tensorflow as tf
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


def _transpose_last_two(
    g: GraphBuilderExtendedProtocol, name: str, inp: tf.Tensor, tag: str
) -> str:
    """Emit an ONNX ``Transpose`` node that swaps the last two dimensions of *inp*.

    :param g: the active :class:`~yobx.xbuilder.GraphBuilder` instance used to
        emit ONNX nodes.
    :param name: base name for the new ONNX node (typically the TF op name).
    :param inp: the TF input tensor whose last two axes are to be swapped;
        its ``.shape`` is inspected to build a rank-aware permutation.
    :param tag: short suffix appended to *name* to form a unique node name
        (e.g. ``"tA"`` or ``"tB"``).
    :returns: the name of the new ONNX tensor produced by the ``Transpose``
        node.
    :raises ValueError: if *inp* has fewer than 2 dimensions.

    The permutation keeps all leading batch axes in place and only swaps
    the last two axes, e.g. for rank 3 it produces ``perm=[0, 2, 1]``.
    """
    rank = len(inp.shape)
    if rank < 2:
        raise ValueError(f"Cannot transpose last two dims of a tensor with rank {rank}.")
    perm = list(range(rank))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return g.op.Transpose(inp.name, perm=perm, name=f"{name}_{tag}")


@register_tf_op_converter(("MatMul", "BatchMatMulV2", "BatchMatMul"))
def convert_matmul(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``MatMul`` / ``BatchMatMulV2`` / ``BatchMatMul`` → ONNX ``MatMul``.

    :param g: the active :class:`~yobx.xbuilder.GraphBuilder` instance.
    :param sts: metadata dictionary (unused, always ``{}``).
    :param outputs: list of pre-allocated output tensor names; ``outputs[0]``
        receives the result of the multiplication.
    :param op: the TF ``MatMul``, ``BatchMatMulV2``, or ``BatchMatMul``
        operation.  Its first two inputs are the left and right operands.
    :returns: the name of the primary output tensor.

    TF's transpose flags (``transpose_a`` / ``transpose_b``) and adjoint flags
    (``adj_x`` / ``adj_y`` used by ``BatchMatMulV2``) are honoured by inserting
    ONNX ``Transpose`` nodes when needed.  For real-valued tensors, adjoint is
    equivalent to transpose.

    The permutation used for transposition is always rank-aware: it swaps only
    the last two axes so that arbitrary batch dimensions are left intact.  See
    :func:`_transpose_last_two` for details.
    """
    attr_names = {attr.name for attr in op.op_def.attr} if op.op_def else set()

    def _is_set(attr_a: str, attr_b: str) -> bool:
        """Return True when either attribute alias is present and truthy."""
        for attr in (attr_a, attr_b):
            if attr in attr_names:
                try:
                    if op.get_attr(attr):
                        return True
                except (ValueError, tf.errors.NotFoundError):
                    pass
        return False

    a = (
        _transpose_last_two(g, op.name, op.inputs[0], "tA")
        if _is_set("transpose_a", "adj_x")
        else op.inputs[0].name
    )
    b = (
        _transpose_last_two(g, op.name, op.inputs[1], "tB")
        if _is_set("transpose_b", "adj_y")
        else op.inputs[1].name
    )
    return g.op.MatMul(a, b, outputs=outputs, name=op.name)
