"""
Per-op FLOPs (floating-point operation) estimators for ONNX nodes.

The public entry point is :func:`estimate_node_flops`.  All per-op helper
functions are private (``_flops_*``) and dispatched via :data:`_OP_HANDLERS`.

All estimators use the symbolic-dimension helpers from
:mod:`yobx.xexpressions.operations` so that symbolic (dynamic) dimensions are
handled correctly.  When the shapes are partially or fully unknown the
estimators return ``None``.

Integration with :class:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder`:
use :meth:`BasicShapeBuilder.estimate_node_flops` to estimate the cost of a
node using the shapes already inferred during a
:meth:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder.run_model` call.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import onnx

from ..xexpressions.operations import dim_add, dim_div, dim_mul, dim_multi_mul, DIM_TYPE

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Maps a tensor name → shape tuple (int for static dims, str for symbolic).
_ShapeFn = Callable[[str], Optional[Tuple[DIM_TYPE, ...]]]

#: Maps a tensor name → integer values of a 1-D integer constant tensor
#: (shape-spec literals, e.g. the second input of a Reshape node).
_LiteralFn = Callable[[str], Optional[Tuple[DIM_TYPE, ...]]]

#: Signature of a per-op FLOPs estimator.
_FlopsHandler = Callable[[onnx.NodeProto, "_ShapeFn", "_LiteralFn"], Optional[DIM_TYPE]]  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Op-type sets used to group estimators
# ---------------------------------------------------------------------------

_ELEMENTWISE_UNARY_OPS: frozenset = frozenset(
    {
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BitShift",
        "Ceil",
        "Celu",
        "Cos",
        "Cosh",
        "Elu",
        "Erf",
        "Exp",
        "Floor",
        "HardSigmoid",
        "HardSwish",
        "LeakyRelu",
        "Log",
        "Mish",
        "Neg",
        "Not",
        "Relu",
        "Round",
        "Selu",
        "Shrink",
        "Sign",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Tan",
        "Tanh",
        "ThresholdedRelu",
    }
)

_ELEMENTWISE_BINARY_OPS: frozenset = frozenset(
    {
        "Add",
        "And",
        "BitAnd",
        "BitOr",
        "BitXor",
        "Div",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "Mod",
        "Mul",
        "Or",
        "Pow",
        "PRelu",
        "Sub",
        "Xor",
    }
)

_NORMALIZATION_OPS: frozenset = frozenset(
    {"LayerNormalization", "GroupNormalization", "InstanceNormalization"}
)

_REDUCE_OPS: frozenset = frozenset(
    {
        "ReduceMax",
        "ReduceMin",
        "ReduceMean",
        "ReduceSum",
        "ReduceProd",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceSumSquare",
    }
)

_ZERO_COST_OPS: frozenset = frozenset({"Identity"})

_DATA_MOVEMENT_OPS: frozenset = frozenset(
    {
        "Cast",
        "CastLike",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Expand",
        "Flatten",
        "Gather",
        "GatherElements",
        "GatherND",
        "OneHot",
        "Pad",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Slice",
        "Split",
        "Tile",
        "Transpose",
    }
)

_RANK_COST_OPS: frozenset = frozenset({"Reshape", "Shape", "Squeeze", "Unsqueeze"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_attribute_value(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    """Returns the value of an attribute, or *default* if missing."""
    for att in node.attribute:
        if att.name != name:
            continue
        if att.type == onnx.AttributeProto.INT:
            return att.i
        if att.type == onnx.AttributeProto.INTS:
            return list(att.ints)
        if att.type == onnx.AttributeProto.FLOAT:
            return att.f
        if att.type == onnx.AttributeProto.FLOATS:
            return list(att.floats)
        if att.type == onnx.AttributeProto.STRING:
            return att.s
    return default


def _literal_size(shape: Optional[Tuple]) -> Optional[DIM_TYPE]:
    """Returns the number of elements in a static shape, or None if dynamic."""
    if shape is None:
        return None
    if all(isinstance(a, int) for a in shape):
        return int(np.prod(shape))
    return dim_multi_mul(*shape)


def _resolve_shape(name: str, shape_fn: _ShapeFn, literal_fn: _LiteralFn) -> Optional[Tuple]:
    """
    Returns the shape of *name* using *shape_fn* first, then *literal_fn* as a
    fallback.  *literal_fn* is consulted when the tensor is a 1-D integer
    constant whose *values* encode a shape specification (e.g. the shape input
    of a Reshape node).
    """
    sh = shape_fn(name)
    if sh is not None:
        return sh
    return literal_fn(name)


# ---------------------------------------------------------------------------
# Per-op-type FLOPs estimator functions
# Each function has the signature:
#   (node, shape_fn, literal_fn) -> Optional[DIM_TYPE]
# ---------------------------------------------------------------------------


def _flops_elementwise_unary(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """1 FLOPs per output element (unary element-wise ops)."""
    if node.output:
        return _literal_size(_resolve_shape(node.output[0], shape_fn, literal_fn))
    return None


def _flops_elementwise_binary(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """1 FLOPs per output element (binary element-wise ops)."""
    return _literal_size(_resolve_shape(node.output[0], shape_fn, literal_fn))


def _flops_sigmoid(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """exp + add + div ≈ 3 FLOPs per output element."""
    s = _literal_size(_resolve_shape(node.output[0], shape_fn, literal_fn))
    return None if s is None else dim_mul(3, s)


def _flops_softmax(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """exp + sum + div ≈ 3 FLOPs per output element."""
    s = _literal_size(_resolve_shape(node.output[0], shape_fn, literal_fn))
    return None if s is None else dim_mul(3, s)


def _flops_matmul(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """2 * batch * M * K * N FLOPs for (batched) matrix multiplication."""
    if len(node.input) < 2:
        return None
    a = _resolve_shape(node.input[0], shape_fn, literal_fn)
    b = _resolve_shape(node.input[1], shape_fn, literal_fn)
    if a is None or b is None:
        return None
    if len(a) < 2 or len(b) < 2:
        return None
    M, K = a[-2], a[-1]
    _K2, N = b[-2], b[-1]
    return dim_multi_mul(*[2, *a[:-2], M, K, N])


def _flops_gemm(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """2*M*K*N + M*N FLOPs for Gemm (alpha*A@B + beta*C)."""
    if len(node.input) < 2:
        return None
    a = _resolve_shape(node.input[0], shape_fn, literal_fn)
    b = _resolve_shape(node.input[1], shape_fn, literal_fn)
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return None
    trans_a = int(_get_attribute_value(node, "transA", 0))
    trans_b = int(_get_attribute_value(node, "transB", 0))
    M, K = (a[1], a[0]) if trans_a else (a[0], a[1])
    _K2, N = (b[1], b[0]) if trans_b else (b[0], b[1])
    return dim_add(dim_multi_mul(2, M, K, N), dim_mul(M, N))


def _flops_conv(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """2 * N * C_out * C_in_per_group * kernel_size * spatial_out FLOPs."""
    if not node.output:
        return None
    out_shape = _resolve_shape(node.output[0], shape_fn, literal_fn)
    if out_shape is None or len(out_shape) < 3:
        return None
    in_shape = _resolve_shape(node.input[0], shape_fn, literal_fn) if node.input else None
    if in_shape is None or len(in_shape) < 2:
        return None
    w_shape = (
        _resolve_shape(node.input[1], shape_fn, literal_fn)
        if len(node.input) >= 2 and node.input[1]
        else None
    )
    if w_shape is None or len(w_shape) < 3:
        return None
    N = out_shape[0]
    C_out = out_shape[1]
    spatial_out = out_shape[2:]
    kernel = w_shape[2:]
    C_in_per_group = w_shape[1]
    spatial_out_size = dim_multi_mul(*spatial_out) if spatial_out else 1
    kernel_size = dim_multi_mul(*kernel) if kernel else 1
    return dim_multi_mul(2, N, C_out, C_in_per_group, kernel_size, spatial_out_size)


def _flops_pool(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """N * C * spatial_out * kernel_size FLOPs for windowed pooling."""
    out_shape = _resolve_shape(node.output[0], shape_fn, literal_fn)
    if out_shape is None or len(out_shape) < 3:
        return None
    N = out_shape[0]
    C = out_shape[1]
    spatial_out = out_shape[2:]
    spatial_out_size = dim_multi_mul(*spatial_out) if spatial_out else 1
    kernel_shape = _get_attribute_value(node, "kernel_shape", None)
    kernel_size = dim_multi_mul(*kernel_shape) if kernel_shape else 1
    return dim_multi_mul(N, C, spatial_out_size, kernel_size)


def _flops_global_pool(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """N * C * spatial FLOPs for global pooling (reduce over spatial dims)."""
    if not node.input:
        return None
    in_shape = _resolve_shape(node.input[0], shape_fn, literal_fn)
    if in_shape is None or len(in_shape) < 3:
        return None
    N = in_shape[0]
    C = in_shape[1]
    spatial = in_shape[2:]
    spatial_size = dim_multi_mul(*spatial) if spatial else 1
    return dim_multi_mul(N, C, spatial_size)


def _flops_batch_norm(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """mean + var + normalize ≈ 2 FLOPs per output element."""
    if not node.output:
        return None
    s = _literal_size(_resolve_shape(node.output[0], shape_fn, literal_fn))
    return None if s is None else dim_mul(2, s)


def _flops_layer_norm(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """mean + var + sub + div + scale + bias ≈ 6 FLOPs per output element."""
    if not node.output:
        return None
    s = _literal_size(_resolve_shape(node.output[0], shape_fn, literal_fn))
    return None if s is None else dim_mul(6, s)


def _flops_reduce(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """Input element count FLOPs for reduction ops."""
    if not node.input:
        return None
    in_shape = _resolve_shape(node.input[0], shape_fn, literal_fn)
    return _literal_size(in_shape)


def _flops_lstm(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """2 * seq * batch * (input_size + hidden) * 4*hidden FLOPs."""
    if len(node.input) < 2:
        return None
    x_shape = _resolve_shape(node.input[0], shape_fn, literal_fn)  # [seq, batch, input_size]
    w_shape = _resolve_shape(node.input[1], shape_fn, literal_fn)  # [num_dir, 4*hidden, input]
    if x_shape is None or w_shape is None:
        return None
    if len(x_shape) < 3 or len(w_shape) < 3:
        return None
    seq, batch, input_size = x_shape[0], x_shape[1], x_shape[2]
    hidden_4 = w_shape[1]
    hidden = dim_div(hidden_4, 4)
    return dim_multi_mul(2, seq, batch, dim_add(input_size, hidden), hidden_4)


def _flops_gru(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """2 * seq * batch * (input_size + hidden) * 3*hidden FLOPs."""
    if len(node.input) < 2:
        return None
    x_shape = _resolve_shape(node.input[0], shape_fn, literal_fn)
    w_shape = _resolve_shape(node.input[1], shape_fn, literal_fn)
    if x_shape is None or w_shape is None:
        return None
    if len(x_shape) < 3 or len(w_shape) < 3:
        return None
    seq, batch, input_size = x_shape[0], x_shape[1], x_shape[2]
    hidden_3 = w_shape[1]
    hidden = dim_div(hidden_3, 3)
    return dim_multi_mul(2, seq, batch, dim_add(input_size, hidden), hidden_3)


def _flops_rnn(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """2 * seq * batch * (input_size + hidden) * hidden FLOPs."""
    if len(node.input) < 2:
        return None
    x_shape = _resolve_shape(node.input[0], shape_fn, literal_fn)
    w_shape = _resolve_shape(node.input[1], shape_fn, literal_fn)
    if x_shape is None or w_shape is None:
        return None
    if len(x_shape) < 3 or len(w_shape) < 3:
        return None
    seq, batch, input_size = x_shape[0], x_shape[1], x_shape[2]
    hidden = w_shape[1]
    return dim_multi_mul(2, seq, batch, dim_add(input_size, hidden), hidden)


def _flops_zero_cost(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> DIM_TYPE:
    """Identity op: 0 FLOPs."""
    return 0


def _flops_data_movement(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """Data movement ops: element count of the first output (or first input)."""
    if node.output:
        s = _literal_size(_resolve_shape(node.output[0], shape_fn, literal_fn))
        if s is not None:
            return s
    if node.input:
        return _literal_size(_resolve_shape(node.input[0], shape_fn, literal_fn))
    return None


def _flops_rank_cost(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """Shape-manipulation ops: cost = rank (number of dimensions) of the output.

    For ``Shape``, the relevant rank is that of the *input* tensor (the op reads
    one value per dimension).  For ``Reshape``, ``Squeeze``, and ``Unsqueeze``
    the rank of the output shape is used.
    """
    if node.op_type == "Shape":
        # Shape reads one value per input dimension.
        if node.input:
            sh = _resolve_shape(node.input[0], shape_fn, literal_fn)
            if sh is not None:
                return len(sh)
        return None
    # Reshape / Squeeze / Unsqueeze: rank of output.
    if node.output:
        sh = _resolve_shape(node.output[0], shape_fn, literal_fn)
        if sh is not None:
            return len(sh)
    return None


# ---------------------------------------------------------------------------
# Dispatcher: maps op_type → handler function
# ---------------------------------------------------------------------------

_OP_HANDLERS: Dict[str, _FlopsHandler] = {}

for _op in _ELEMENTWISE_UNARY_OPS:
    _OP_HANDLERS[_op] = _flops_elementwise_unary
for _op in _ELEMENTWISE_BINARY_OPS:
    _OP_HANDLERS[_op] = _flops_elementwise_binary
for _op in _NORMALIZATION_OPS:
    _OP_HANDLERS[_op] = _flops_layer_norm
for _op in _REDUCE_OPS:
    _OP_HANDLERS[_op] = _flops_reduce
for _op in _ZERO_COST_OPS:
    _OP_HANDLERS[_op] = _flops_zero_cost
for _op in _DATA_MOVEMENT_OPS:
    _OP_HANDLERS[_op] = _flops_data_movement
for _op in _RANK_COST_OPS:
    _OP_HANDLERS[_op] = _flops_rank_cost

_OP_HANDLERS.update(
    {
        "Sigmoid": _flops_sigmoid,
        "Softmax": _flops_softmax,
        "LogSoftmax": _flops_softmax,
        "MatMul": _flops_matmul,
        "Gemm": _flops_gemm,
        "Conv": _flops_conv,
        "ConvTranspose": _flops_conv,
        "MaxPool": _flops_pool,
        "AveragePool": _flops_pool,
        "GlobalAveragePool": _flops_global_pool,
        "GlobalMaxPool": _flops_global_pool,
        "BatchNormalization": _flops_batch_norm,
        "LSTM": _flops_lstm,
        "GRU": _flops_gru,
        "RNN": _flops_rnn,
    }
)


def estimate_node_flops(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """
    Estimates the number of floating-point operations for a single ONNX node.

    Returns ``None`` when the shapes are not fully known (dynamic shapes) or the
    ``op_type`` is not covered.

    :param node: ONNX node
    :param shape_fn: callable mapping tensor name → shape tuple (from shape inference)
    :param literal_fn: callable mapping tensor name → int-value tuple for 1-D integer
        constant tensors (shape specification tensors); used as a fallback when
        *shape_fn* cannot resolve a shape
    :return: estimated number of FLOPs, or ``None``
    """
    handler = _OP_HANDLERS.get(node.op_type)
    if handler is None:
        return None
    return handler(node, shape_fn, literal_fn)


def list_op_cost_formulas() -> Dict[str, str]:
    """
    Returns a mapping from each supported ONNX ``op_type`` to a human-readable
    description of the FLOPs formula used by :func:`estimate_node_flops`.

    The descriptions are taken directly from the docstrings of the per-op handler
    functions registered in :data:`_OP_HANDLERS`.  Operators that share the same
    handler (e.g. all element-wise unary ops) therefore share the same description.

    :return: ``{op_type: formula_description}`` sorted alphabetically by ``op_type``.
    """
    result: Dict[str, str] = {}
    for op_type, handler in _OP_HANDLERS.items():
        doc = (handler.__doc__ or "").strip().splitlines()[0].rstrip(".")
        result[op_type] = doc
    return dict(sorted(result.items()))
