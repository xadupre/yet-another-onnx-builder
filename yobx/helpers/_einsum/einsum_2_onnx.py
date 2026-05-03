"""
Direct ONNX graph builder for 2-input einsum equations.

This module provides :func:`decompose_einsum_2inputs`, a completely
independent implementation that converts a 2-operand einsum equation
directly into a sequence of basic ONNX operators without relying on the
:class:`~yobx.helpers._einsum.EinsumSubOp` /
:class:`~yobx.helpers._einsum.GraphEinsumSubOp` graph representation used
by the ``"simple"`` and ``"numpy"`` strategies.

Algorithm
---------
For a 2-input equation ``lhs0,lhs1->rhs`` the algorithm classifies every
index letter into one of four roles:

* **batch** – present in both inputs *and* the output.  These dimensions
  are kept as leading batch dimensions throughout.
* **contract** – present in both inputs but *absent* from the output.
  These are the summation (dot-product) axes.
* **left** – present only in the first input (and therefore in the
  output).
* **right** – present only in the second input (and therefore in the
  output).

It then builds the following sequence of ONNX nodes:

1. **Transpose** each input (if required) so their axes appear in the
   canonical order ``[batch…, left/right…, contract…]``.
2. **Reshape** each transposed input into a 3-D tensor
   ``[batch_prod, free_prod, contract_prod]`` using ``Shape`` /
   ``Gather`` / ``ReduceProd`` nodes so the graph is compatible with
   variable-length (dynamic) dimensions.
3. **MatMul** ``[batch_prod, left_prod, contract_prod]`` ×
   ``[batch_prod, contract_prod, right_prod]``
   → ``[batch_prod, left_prod, right_prod]``.
4. **Reshape** the result back to ``[batch_dims…, left_dims…,
   right_dims…]``.
5. **Transpose** (if required) to the requested output subscript order.

Empty groups (e.g. no batch dimensions) are handled by using a constant
``1`` as the corresponding product so the 3-D MatMul remains well-formed.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_2input_equation(equation: str) -> Tuple[str, str, str]:
    """
    Parses a 2-input einsum equation into ``(lhs0, lhs1, rhs)``.

    :param equation: einsum equation string, e.g. ``"ij,jk->ik"``.
    :return: triple ``(lhs0, lhs1, rhs)``.
    :raises ValueError: if the equation does not have exactly 2 inputs and
        one output separated by ``"->"``.
    """
    parts = equation.replace(" ", "").split("->")
    if len(parts) != 2 or not parts[0]:
        raise ValueError(
            f"Equation {equation!r} must have the form 'ij,jk->ik' "
            "with exactly two inputs and one output."
        )
    lhs_parts = parts[0].split(",")
    if len(lhs_parts) != 2:
        raise ValueError(
            f"Equation {equation!r} must have exactly 2 input operands, got {len(lhs_parts)}."
        )
    return lhs_parts[0], lhs_parts[1], parts[1]


def is_identity_perm(perm: List[int]) -> bool:
    """Returns ``True`` when *perm* is the identity permutation."""
    return perm == list(range(len(perm)))


def const_int64(value: List[int], name: str) -> onnx.TensorProto:
    """Creates an int64 initializer tensor."""
    return onh.from_array(numpy.array(value, dtype=numpy.int64), name=name)


class EinsumBuilder:
    """
    Stateful helper that accumulates ONNX nodes and initializers while
    generating unique names.
    """

    def __init__(self, opset: int):
        self.opset = opset
        self.nodes: List[onnx.NodeProto] = []
        self.initializers: List[onnx.TensorProto] = []
        self._counter = 0

    def _uid(self) -> int:
        self._counter += 1
        return self._counter

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    def add_init(self, tensor: onnx.TensorProto) -> str:
        """Registers *tensor* as an initializer and returns its name."""
        self.initializers.append(tensor)
        return tensor.name

    def add_node(self, node: onnx.NodeProto) -> str:
        """Registers *node* and returns the name of its first output."""
        self.nodes.append(node)
        return node.output[0]

    # ------------------------------------------------------------------
    # Building-block operations
    # ------------------------------------------------------------------

    def const_1d(self, values: List[int], prefix: str = "c") -> str:
        """Emits a 1-D int64 constant and returns its name."""
        name = f"{prefix}_{self._uid()}"
        return self.add_init(const_int64(values, name))

    def identity(self, inp: str, prefix: str = "id") -> str:
        """Emits an Identity node (used as a no-op rename)."""
        out = f"{prefix}_{self._uid()}"
        self.nodes.append(oh.make_node("Identity", [inp], [out]))
        return out

    def transpose(self, inp: str, perm: List[int], prefix: str = "tr") -> str:
        """Emits a Transpose node if *perm* is not the identity."""
        if is_identity_perm(perm):
            return inp
        out = f"{prefix}_{self._uid()}"
        self.add_node(oh.make_node("Transpose", [inp], [out], perm=perm))
        return out

    def shape(self, inp: str, prefix: str = "shp") -> str:
        """Emits a Shape node and returns its output name."""
        out = f"{prefix}_{self._uid()}"
        self.add_node(oh.make_node("Shape", [inp], [out]))
        return out

    def gather_dims(self, shape_name: str, indices: List[int], prefix: str) -> str:
        """
        Gathers specific dimension values from a shape vector.

        Returns a 1-D int64 tensor of length ``len(indices)`` holding the
        gathered dimension values, or a constant ``[1]`` when *indices* is
        empty.
        """
        if not indices:
            return self.const_1d([1], prefix + "_one")
        idx_name = self.const_1d(indices, prefix + "_idx")
        out = f"{prefix}_gathered_{self._uid()}"
        self.add_node(oh.make_node("Gather", [shape_name, idx_name], [out], axis=0))
        return out

    def reduce_prod_1d(self, inp: str, prefix: str) -> str:
        """
        Reduces a 1-D int64 tensor to a scalar wrapped in a 1-D tensor
        ``[product]`` via ``ReduceProd``.
        """
        out = f"{prefix}_prod_{self._uid()}"
        self.add_node(oh.make_node("ReduceProd", [inp], [out], keepdims=1))
        return out

    def dim_product(self, shape_name: str, indices: List[int], prefix: str) -> str:
        """
        Computes the product of shape dimensions at *indices* as a 1-D
        tensor of length 1.  Returns constant ``[1]`` when *indices* is
        empty.
        """
        gathered = self.gather_dims(shape_name, indices, prefix)
        if not indices:
            return gathered  # already the constant [1]
        if len(indices) == 1:
            # Single element – already 1-D of length 1; no ReduceProd needed.
            return gathered
        return self.reduce_prod_1d(gathered, prefix)

    def reshape(self, inp: str, shape_inp: str, prefix: str = "resh") -> str:
        """Emits a Reshape node."""
        out = f"{prefix}_{self._uid()}"
        self.add_node(oh.make_node("Reshape", [inp, shape_inp], [out]))
        return out

    def concat_shapes(self, parts: List[str], prefix: str = "cat") -> str:
        """Concatenates a list of 1-D shape tensors along axis 0."""
        if len(parts) == 1:
            return parts[0]
        out = f"{prefix}_{self._uid()}"
        self.add_node(oh.make_node("Concat", parts, [out], axis=0))
        return out

    def matmul(self, a: str, b: str, prefix: str = "mm") -> str:
        """Emits a MatMul node."""
        out = f"{prefix}_{self._uid()}"
        self.add_node(oh.make_node("MatMul", [a, b], [out]))
        return out

    @staticmethod
    def make_value_info(
        name: str, elem_type: int, shape: Optional[Sequence[Optional[Union[int, str]]]]
    ) -> onnx.ValueInfoProto:
        """
        Builds an ONNX ``ValueInfoProto`` for a tensor input.

        :param name: tensor name.
        :param elem_type: ONNX element type (e.g. ``onnx.TensorProto.FLOAT``).
        :param shape: optional list of dimension sizes.  Each element may be an
            integer (fixed size), a string (symbolic name), or ``None`` (dynamic).
            Pass ``None`` for a fully unranked input.
        :return: :class:`onnx.ValueInfoProto`.
        """
        if shape is None:
            return oh.make_tensor_value_info(name, elem_type, None)
        onnx_dims: List[Optional[Union[int, str]]] = list(shape)
        return oh.make_tensor_value_info(name, elem_type, onnx_dims)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose_einsum_2inputs(
    equation: str,
    shape0: Optional[Sequence[Optional[Union[int, str]]]] = None,
    shape1: Optional[Sequence[Optional[Union[int, str]]]] = None,
    name0: str = "X0",
    name1: str = "X1",
    output_name: str = "Z",
    dtype: int = onnx.TensorProto.FLOAT,
    opset: Optional[int] = None,
) -> onnx.ModelProto:
    """
    Decomposes a 2-input einsum equation directly into basic ONNX operators.

    This is a fully self-contained implementation that does **not** depend
    on the :class:`~yobx.helpers._einsum.EinsumSubOp` /
    :class:`~yobx.helpers._einsum.GraphEinsumSubOp` framework.

    :param equation: einsum equation string with exactly two input operands
        and an explicit output, e.g. ``"ij,jk->ik"`` or ``"bij,bjk->bik"``.
    :param shape0: optional shape of the first input.  Each element may be
        an integer (fixed size), a string (symbolic name), or ``None``
        (fully dynamic).  When omitted the input has no shape annotation.
    :param shape1: optional shape of the second input (same convention).
    :param name0: name used for the first graph input (default ``"X0"``).
    :param name1: name used for the second graph input (default ``"X1"``).
    :param output_name: name used for the graph output (default ``"Z"``).
    :param dtype: ONNX element type for all inputs and the output
        (default :data:`onnx.TensorProto.FLOAT`).
    :param opset: ONNX opset version; defaults to the current ONNX opset
        capped at 18.
    :return: :class:`onnx.ModelProto` that computes
        ``numpy.einsum(equation, X0, X1)``.
    :raises ValueError: if *equation* does not have exactly two inputs or
        contains letters in the output that do not appear in any input.

    Example::

        import numpy as np
        import onnxruntime
        from yobx.helpers._einsum.einsum_2_onnx import decompose_einsum_2inputs

        model = decompose_einsum_2inputs("ij,jk->ik", (3, 4), (4, 5))
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(4, 5).astype(np.float32)
        (result,) = sess.run(None, {"X0": a, "X1": b})
        assert np.allclose(result, np.einsum("ij,jk->ik", a, b), atol=1e-5)
    """
    if opset is None:
        opset = min(18, onnx.defs.onnx_opset_version())

    lhs0, lhs1, rhs = parse_2input_equation(equation)

    # ------------------------------------------------------------------
    # Classify index letters into four disjoint roles.
    # ------------------------------------------------------------------
    set0 = set(lhs0)
    set1 = set(lhs1)
    set_out = set(rhs)

    # Letters must come from one of these four categories; validate rhs.
    valid_out = (set0 & set1 & set_out) | (set0 - set1) | (set1 - set0)
    extra = set_out - valid_out
    if extra:
        raise ValueError(
            f"Output letters {sorted(extra)!r} in equation {equation!r} "
            "do not appear in any input operand."
        )

    # sorted() for deterministic permutations
    batch = sorted(set0 & set1 & set_out)  # present in both + output
    contract = sorted(set0 & set1 - set_out)  # present in both, reduced out
    left = sorted(set0 - set1)  # first input only
    right = sorted(set1 - set0)  # second input only

    nb = len(batch)
    nl = len(left)
    nc = len(contract)
    nr = len(right)

    # ------------------------------------------------------------------
    # Compute permutations.
    # ------------------------------------------------------------------
    # X0 target order: [batch…, left…, contract…]
    target0 = batch + left + contract
    perm0 = [lhs0.index(c) for c in target0]

    # X1 target order: [batch…, contract…, right…]
    target1 = batch + contract + right
    perm1 = [lhs1.index(c) for c in target1]

    # Output order after MatMul + reshape is [batch…, left…, right…].
    # Permutation to reach rhs order.
    combined = batch + left + right
    perm_out = [combined.index(c) for c in rhs]

    # ------------------------------------------------------------------
    # Build the ONNX graph.
    # ------------------------------------------------------------------
    bld = EinsumBuilder(opset)

    # Step 1 – Transpose inputs to canonical order.
    x0t = bld.transpose(name0, perm0, prefix="x0t")
    x1t = bld.transpose(name1, perm1, prefix="x1t")

    # Step 2 – Compute 3-D reshape targets using dynamic Shape nodes.
    #   X0 transposed: [batch…, left…, contract…]
    #   X1 transposed: [batch…, contract…, right…]
    shp0 = bld.shape(x0t, prefix="shp0")
    shp1 = bld.shape(x1t, prefix="shp1")

    # Products of each group (all are 1-D tensors of length 1).
    batch_prod = bld.dim_product(shp0, list(range(nb)), "batch_prod")
    left_prod = bld.dim_product(shp0, list(range(nb, nb + nl)), "left_prod")
    contract_prod = bld.dim_product(shp0, list(range(nb + nl, nb + nl + nc)), "contract_prod")
    right_prod = bld.dim_product(shp1, list(range(nb + nc, nb + nc + nr)), "right_prod")

    # Step 3 – Reshape to 3-D.
    x0_3d_shape = bld.concat_shapes([batch_prod, left_prod, contract_prod], "x0_3d_shape")
    x0_3d = bld.reshape(x0t, x0_3d_shape, "x0_3d")

    x1_3d_shape = bld.concat_shapes([batch_prod, contract_prod, right_prod], "x1_3d_shape")
    x1_3d = bld.reshape(x1t, x1_3d_shape, "x1_3d")

    # Step 4 – MatMul: [A, B, C] × [A, C, D] → [A, B, D]
    mm = bld.matmul(x0_3d, x1_3d, "mm")

    # Step 5 – Rebuild full-rank output shape from individual dim values.
    out_parts: List[str] = []
    if nb > 0:
        out_parts.append(bld.gather_dims(shp0, list(range(nb)), "out_batch"))
    if nl > 0:
        out_parts.append(bld.gather_dims(shp0, list(range(nb, nb + nl)), "out_left"))
    if nr > 0:
        out_parts.append(bld.gather_dims(shp1, list(range(nb + nc, nb + nc + nr)), "out_right"))

    if not out_parts:
        # Scalar output: reshape [1,1,1] → []
        scalar_shape = bld.const_1d([], "scalar_shape")
        result_full = bld.reshape(mm, scalar_shape, "result_scalar")
    else:
        out_shape = bld.concat_shapes(out_parts, "out_shape")
        result_full = bld.reshape(mm, out_shape, "result_full")

    # Step 6 – Transpose to the requested output order (if needed).
    final = bld.transpose(result_full, perm_out, prefix="result")

    # Rename the final node output to ``output_name``.
    if final != output_name:
        bld.nodes.append(oh.make_node("Identity", [final], [output_name]))

    # ------------------------------------------------------------------
    # Assemble the ONNX ModelProto.
    # ------------------------------------------------------------------
    graph = oh.make_graph(
        bld.nodes,
        "einsum_2inputs",
        [bld.make_value_info(name0, dtype, shape0), bld.make_value_info(name1, dtype, shape1)],
        [oh.make_tensor_value_info(output_name, dtype, None)],
        initializer=bld.initializers,
    )

    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    model.ir_version = onnx.IR_VERSION
    return model
