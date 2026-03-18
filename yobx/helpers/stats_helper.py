"""
Functions to compute statistics on an ONNX model such as number of nodes
per op_type and estimation of computational cost.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from ..xexpressions.operations import dim_add, dim_div, dim_mul, dim_multi_mul, DIM_TYPE
from ..typing import GraphBuilderExtendedProtocol

# Type aliases for the shape and literal lookup functions passed to per-op helpers.
#
# _ShapeFn: maps a tensor name to its shape as a tuple of dimensions.
#   Each dimension is an int for static shapes or a str for symbolic (dynamic) dims.
#   Returns None when the shape is completely unknown.
#
# _LiteralFn: maps a tensor name to the *integer values* stored in a 1-D integer
#   constant tensor (a shape specification literal such as the second input of
#   Reshape).  Returns None when the tensor is not a known 1-D int constant.
_ShapeFn = Callable[[str], Optional[Tuple[DIM_TYPE, ...]]]
_LiteralFn = Callable[[str], Optional[Tuple[DIM_TYPE, ...]]]

# Type alias for an op-level FLOPs estimator function.
_FlopsHandler = Callable[
    [onnx.NodeProto, "_ShapeFn", "_LiteralFn"], Optional[DIM_TYPE]  # type: ignore[type-arg]
]


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
#   (node, shape_fn, literal_fn) -> Optional[int]
# ---------------------------------------------------------------------------

# Op sets for element-wise ops that cost 1 FLOPs per output element
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

_ZERO_COST_OPS: frozenset = frozenset(
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
        "Identity",
        "OneHot",
        "Pad",
        "Reshape",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Shape",
        "Slice",
        "Split",
        "Squeeze",
        "Tile",
        "Transpose",
        "Unsqueeze",
    }
)


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
    """Data movement ops: 0 FLOPs."""
    return 0


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


def _estimate_node_flops(
    node: onnx.NodeProto, shape_fn: _ShapeFn, literal_fn: _LiteralFn
) -> Optional[DIM_TYPE]:
    """
    Estimates the number of floating-point operations for a single ONNX node.

    Returns None when the shapes are not fully known (dynamic shapes) or the
    op_type is not covered.

    :param node: ONNX node
    :param shape_fn: callable mapping tensor name → shape tuple (from shape inference)
    :param literal_fn: callable mapping tensor name → int-value tuple for 1-D integer
        constant tensors (shape specification tensors); used as a fallback when
        *shape_fn* cannot resolve a shape
    :return: estimated number of FLOPs, or None
    """
    handler = _OP_HANDLERS.get(node.op_type)
    if handler is None:
        return None
    return handler(node, shape_fn, literal_fn)


class ModelStatistics:
    """
    Computes statistics on an ONNX model, including node counts per op_type
    and estimated FLOPs.

    *model* may be either an :class:`onnx.ModelProto` or a
    :class:`~yobx.typing.GraphBuilderExtendedProtocol` instance.  When a graph
    builder is provided its already-computed shape information is used directly
    (no second shape-inference pass is run) and the ONNX model is obtained via
    :meth:`~yobx.typing.GraphBuilderProtocol.to_onnx`.

    :param model: ONNX model or graph builder
    :param verbose: verbosity level passed to :class:`BasicShapeBuilder`
        (ignored when *model* is a graph builder)

    Usage::

        stats = ModelStatistics(model).compute()
    """

    def __init__(
        self, model: Union[onnx.ModelProto, GraphBuilderExtendedProtocol], verbose: int = 0
    ) -> None:
        if isinstance(model, GraphBuilderExtendedProtocol):
            self._builder: Any = model
            onnx_model = model.to_onnx()
            if not isinstance(onnx_model, onnx.ModelProto):
                raise TypeError(
                    f"GraphBuilderExtendedProtocol.to_onnx() returned "
                    f"{type(onnx_model).__name__!r}, expected ModelProto"
                )
            self.model: onnx.ModelProto = onnx_model
        else:
            self._builder = None
            self.model = model
        self.verbose = verbose
        self._bs: Any = None
        self._shape_map: Dict[str, Optional[Tuple]] = {}
        self._node_count: Dict[str, int] = {}
        self._flops_by_op: Dict[str, List[Optional[DIM_TYPE]]] = {}
        self._node_stats: List[Dict[str, Any]] = []

    def _collect_names(self, graph: onnx.GraphProto, all_names: set) -> None:
        """Recursively gathers every tensor name referenced in *graph*."""
        for vi in list(graph.input) + list(graph.output):
            all_names.add(vi.name)
        for init in graph.initializer:
            all_names.add(init.name)
        for node in graph.node:
            for n in list(node.input) + list(node.output):
                if n:
                    all_names.add(n)
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    self._collect_names(att.g, all_names)
                elif att.type == onnx.AttributeProto.GRAPHS:
                    for g in att.graphs:
                        self._collect_names(g, all_names)

    def _build_shape_map(self) -> None:
        """Populates :attr:`_shape_map` from the builder (if provided) or by
        running :class:`BasicShapeBuilder` on the stored model."""
        all_names: set = set()
        self._collect_names(self.model.graph, all_names)

        if self._builder is not None:
            # Use shape information already computed by the graph builder.
            for name in all_names:
                if not self._builder.has_shape(name):
                    continue
                sh = self._builder.get_shape(name)
                if all(isinstance(d, int) for d in sh):
                    self._shape_map[name] = tuple(int(d) for d in sh)
                else:
                    self._shape_map[name] = sh
        else:
            from ..xshape import BasicShapeBuilder

            self._bs = BasicShapeBuilder(verbose=self.verbose)
            self._bs.run_model(self.model)

            for name in all_names:
                if not self._bs.has_shape(name):
                    continue
                sh = self._bs.get_shape(name)
                if all(isinstance(d, int) for d in sh):
                    self._shape_map[name] = tuple(int(d) for d in sh)
                else:
                    self._shape_map[name] = sh

    def shape_fn(self, name: str) -> Optional[Tuple]:
        """Returns the inferred shape of *name*, or ``None`` if unknown."""
        return self._shape_map.get(name)

    def literal_fn(self, name: str) -> Optional[Tuple[int, ...]]:
        """
        Returns the integer values stored in a 1-D integer constant tensor
        (e.g. the ``shape`` input of a Reshape node), or ``None`` when *name*
        is not a known constant.
        """
        if not name:
            return None
        if self._builder is not None:
            # Use the builder's constant API via duck typing.
            is_const_fn = getattr(self._builder, "is_constant", None)
            if is_const_fn is not None and not is_const_fn(name):
                return None
            get_fn = getattr(self._builder, "get_computed_constant", None)
            if get_fn is not None:
                try:
                    val = get_fn(name)
                    if val is not None:
                        return tuple(int(v) for v in np.asarray(val).flat)
                except Exception:
                    pass
            return None
        if not self._bs.is_constant(name):
            return None
        val = self._bs.get_constant(name, exc=False, computed_value=True, as_shape=True)
        if val is None:
            return None
        return tuple(int(v) for v in val)  # type: ignore

    def _collect_nodes(self, graph: onnx.GraphProto) -> None:
        """Recursively visits all nodes in *graph* and accumulates statistics."""
        for node in graph.node:
            op = node.op_type
            self._node_count[op] = self._node_count.get(op, 0) + 1

            f = _estimate_node_flops(node, self.shape_fn, self.literal_fn)
            if op not in self._flops_by_op:
                self._flops_by_op[op] = []
            self._flops_by_op[op].append(f)

            input_shapes = [(n, self._shape_map.get(n)) for n in node.input if n]
            output_shapes = [(n, self._shape_map.get(n)) for n in node.output if n]
            self._node_stats.append(
                {
                    "op_type": op,
                    "name": node.name,
                    "inputs": input_shapes,
                    "outputs": output_shapes,
                    "estimated_flops": f,
                }
            )
            # Recurse into subgraphs (e.g. inside If / Loop / Scan)
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    self._collect_nodes(att.g)
                elif att.type == onnx.AttributeProto.GRAPHS:
                    for g in att.graphs:
                        self._collect_nodes(g)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> List[Dict[str, Any]]:
        """
        Runs the full analysis and returns a statistics dictionary.

        :return: dictionary with the following keys:

            - ``"n_nodes"`` – total number of nodes
            - ``"node_count_per_op_type"`` – dict mapping op_type → count
            - ``"total_estimated_flops"`` – total estimated FLOPs (int or None)
            - ``"flops_per_op_type"`` – dict mapping op_type → estimated FLOPs
              (None means the cost could not be estimated for some nodes)
            - ``"node_stats"`` – list of per-node dicts with keys
              ``op_type``, ``name``, ``inputs``, ``outputs``, ``estimated_flops``
        """
        self._build_shape_map()
        self._collect_nodes(self.model.graph)
        return self._node_stats


def model_statistics(
    model: Union[onnx.ModelProto, GraphBuilderExtendedProtocol], verbose: int = 0
) -> List[Dict[str, Any]]:
    """
    Computes statistics on an ONNX model.

    This is a convenience wrapper around :class:`ModelStatistics`.

    :param model: ONNX model or graph builder
    :param verbose: verbosity level
    :return: statistics dictionary — see :meth:`ModelStatistics.compute` for details
    """
    return ModelStatistics(model, verbose=verbose).compute()
