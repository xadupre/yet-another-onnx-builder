"""
Proxy class for tracing numpy operations and converting them to ONNX.

Each :class:`NumpyArray` wraps an ONNX tensor name inside a
:class:`~yobx.xbuilder.GraphBuilder`.  Arithmetic operators, numpy ufuncs and
numpy array-functions performed on a :class:`NumpyArray` are recorded as ONNX
nodes in the underlying graph so that the resulting computation graph can
later be exported to an ONNX model.
"""

from typing import Any, Dict, Optional, Union
import numpy as np

# ---------------------------------------------------------------------------
# Ufunc → ONNX operator mapping
# ---------------------------------------------------------------------------

#: Maps numpy ufuncs to ``(onnx_op_name, n_inputs)`` pairs, or a sentinel
#: string for ufuncs that need special handling.
_UFUNC_TO_ONNX: Dict[Any, Any] = {
    np.add: ("Add", 2),
    np.subtract: ("Sub", 2),
    np.multiply: ("Mul", 2),
    np.true_divide: ("Div", 2),
    np.divide: ("Div", 2),
    np.floor_divide: "floor_divide",  # Floor(Div(a,b))
    np.power: ("Pow", 2),
    np.mod: ("Mod", 2),
    np.fmod: ("Mod", 2),
    np.negative: ("Neg", 1),
    np.absolute: ("Abs", 1),
    np.sqrt: ("Sqrt", 1),
    np.exp: ("Exp", 1),
    np.log: ("Log", 1),
    np.sin: ("Sin", 1),
    np.cos: ("Cos", 1),
    np.tan: ("Tan", 1),
    np.sinh: ("Sinh", 1),
    np.cosh: ("Cosh", 1),
    np.tanh: ("Tanh", 1),
    np.arcsin: ("Asin", 1),
    np.arccos: ("Acos", 1),
    np.arctan: ("Atan", 1),
    np.ceil: ("Ceil", 1),
    np.floor: ("Floor", 1),
    np.sign: ("Sign", 1),
    np.isnan: ("IsNaN", 1),
    np.reciprocal: ("Reciprocal", 1),
    np.matmul: ("MatMul", 2),
    np.greater: ("Greater", 2),
    np.greater_equal: ("GreaterOrEqual", 2),
    np.less: ("Less", 2),
    np.less_equal: ("LessOrEqual", 2),
    np.equal: ("Equal", 2),
    np.not_equal: "not_equal",  # Equal + Not
    np.logical_and: ("And", 2),
    np.logical_or: ("Or", 2),
    np.logical_xor: ("Xor", 2),
    np.logical_not: ("Not", 1),
    np.bitwise_and: ("And", 2),
    np.bitwise_or: ("Or", 2),
    np.bitwise_xor: ("Xor", 2),
    np.invert: ("Not", 1),
    np.maximum: "maximum",  # Max(a, b)
    np.minimum: "minimum",  # Min(a, b)
    np.log1p: "log1p",  # Add(x, 1) + Log
    np.expm1: "expm1",  # Exp(x) - 1
}

# ---------------------------------------------------------------------------
# __array_function__ dispatch table
# ---------------------------------------------------------------------------

#: Dispatch table for numpy array functions (populated by :func:`_implements`).
_HANDLED_FUNCTIONS: Dict[Any, Any] = {}


def _implements(np_function: Any):
    """Decorator that registers an ``__array_function__`` implementation."""

    def decorator(func):
        _HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _scalar_to_array(value: Union[int, float, bool], dtype) -> np.ndarray:
    """Convert a Python scalar to a numpy array with the given *dtype*."""
    np_dtype = dtype if dtype is not None else np.float32
    return np.array(value, dtype=np_dtype)


def _get_g(*args: Any):
    """Return the first :class:`GraphBuilder` found in *args*."""
    for arg in args:
        if isinstance(arg, NumpyArray):
            return arg._g
        if isinstance(arg, (list, tuple)):
            g = _get_g(*arg)
            if g is not None:
                return g
    return None


def _to_onnx_arg(value: Any, g, ref_dtype=None) -> Any:
    """
    Convert *value* to an argument accepted by :class:`~yobx.xbuilder.graph_builder_opset.Opset`.

    * :class:`NumpyArray` → returned as-is (Opset extracts ``.name``)
    * :class:`numpy.ndarray` → returned as-is (Opset converts to initializer)
    * Python scalar (int/float/bool) → converted to numpy scalar
    * ``None`` → returned as-is (Opset converts to ``""`` for optional inputs)
    """
    if isinstance(value, NumpyArray):
        return value
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float, bool)):
        return _scalar_to_array(value, ref_dtype)
    if isinstance(value, np.generic):
        # numpy scalar types (np.float32(1.0), np.int64(3), etc.)
        return np.asarray(value)
    if value is None:
        return None
    raise TypeError(
        f"Cannot convert {type(value).__name__!r} to an ONNX argument; "
        "expected NumpyArray, numpy.ndarray, a Python scalar, or None."
    )


# ---------------------------------------------------------------------------
# NumpyArray
# ---------------------------------------------------------------------------


class NumpyArray:
    """
    Proxy for an ONNX tensor that traces numpy operations as ONNX graph nodes.

    Instances are produced by :func:`~yobx.xtracing.tracing.trace_numpy_to_onnx`
    or directly by the :class:`~sklearn.preprocessing.FunctionTransformer`
    converter when the input array is replaced by a symbolic placeholder.
    Every arithmetic operation, ufunc call, or reduction performed on a
    :class:`NumpyArray` is recorded as an ONNX node in the underlying
    :class:`~yobx.xbuilder.GraphBuilder`.

    The class follows the `Python Array API standard
    <https://data-apis.org/array-api/latest/>`_ and the numpy ``__array_ufunc__``
    / ``__array_function__`` dispatch protocols so that plain numpy code can be
    traced without modification.

    :param name: ONNX tensor name (a string handle in the graph).
    :param graph_builder: the :class:`~yobx.xbuilder.GraphBuilder` that owns
        the graph being built.
    :param dtype: optional numpy dtype for the tensor; used when creating
        scalar constants from Python literals.
    :param shape: optional tensor shape.
    """

    # Tell numpy to use our protocols instead of trying to convert to array.
    __array_priority__ = 20.0

    def __init__(self, name: str, graph_builder, dtype=None, shape=None):
        self._name = name
        self._g = graph_builder
        self._dtype = np.dtype(dtype) if dtype is not None else None
        self._shape = shape

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """ONNX tensor name."""
        return self._name

    @property
    def dtype(self) -> Optional[np.dtype]:
        """Numpy dtype if known."""
        return self._dtype

    @property
    def shape(self):
        """Tensor shape if known."""
        return self._shape

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new(self, name: str) -> "NumpyArray":
        """Return a new :class:`NumpyArray` in the same graph."""
        return NumpyArray(name, self._g)

    def _arg(self, value: Any) -> Any:
        """Convert *value* to an Opset-compatible argument."""
        return _to_onnx_arg(value, self._g, ref_dtype=self._dtype)

    def _binary(self, op: str, a: Any, b: Any) -> "NumpyArray":
        """Emit a binary ONNX operation."""
        res = getattr(self._g.op, op)(
            self._arg(a), self._arg(b), name=self._g.unique_name(op.lower())
        )
        return self._new(res)

    def _unary(self, op: str) -> "NumpyArray":
        """Emit a unary ONNX operation."""
        res = getattr(self._g.op, op)(self._name, name=self._g.unique_name(op.lower()))
        return self._new(res)

    def _reduce(self, onnx_op: str, axis=None, keepdims: bool = False) -> "NumpyArray":
        """Emit a reduction ONNX operation."""
        kd = int(keepdims)
        if axis is None:
            res = getattr(self._g.op, onnx_op)(
                self._name, keepdims=kd, name=self._g.unique_name(onnx_op.lower())
            )
        else:
            axes_arr = np.array([axis] if isinstance(axis, int) else list(axis), dtype=np.int64)
            res = getattr(self._g.op, onnx_op)(
                self._name, axes_arr, keepdims=kd, name=self._g.unique_name(onnx_op.lower())
            )
        return self._new(res)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other) -> "NumpyArray":
        return self._binary("Add", self, other)

    def __radd__(self, other) -> "NumpyArray":
        return self._binary("Add", other, self)

    def __sub__(self, other) -> "NumpyArray":
        return self._binary("Sub", self, other)

    def __rsub__(self, other) -> "NumpyArray":
        return self._binary("Sub", other, self)

    def __mul__(self, other) -> "NumpyArray":
        return self._binary("Mul", self, other)

    def __rmul__(self, other) -> "NumpyArray":
        return self._binary("Mul", other, self)

    def __truediv__(self, other) -> "NumpyArray":
        return self._binary("Div", self, other)

    def __rtruediv__(self, other) -> "NumpyArray":
        return self._binary("Div", other, self)

    def __floordiv__(self, other) -> "NumpyArray":
        div = self._binary("Div", self, other)
        return div._unary("Floor")

    def __rfloordiv__(self, other) -> "NumpyArray":
        div = self._binary("Div", other, self)
        return div._unary("Floor")

    def __mod__(self, other) -> "NumpyArray":
        return self._binary("Mod", self, other)

    def __rmod__(self, other) -> "NumpyArray":
        return self._binary("Mod", other, self)

    def __pow__(self, other) -> "NumpyArray":
        return self._binary("Pow", self, other)

    def __rpow__(self, other) -> "NumpyArray":
        return self._binary("Pow", other, self)

    def __neg__(self) -> "NumpyArray":
        return self._unary("Neg")

    def __abs__(self) -> "NumpyArray":
        return self._unary("Abs")

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------

    def __lt__(self, other) -> "NumpyArray":
        return self._binary("Less", self, other)

    def __le__(self, other) -> "NumpyArray":
        return self._binary("LessOrEqual", self, other)

    def __gt__(self, other) -> "NumpyArray":
        return self._binary("Greater", self, other)

    def __ge__(self, other) -> "NumpyArray":
        return self._binary("GreaterOrEqual", self, other)

    def __eq__(self, other) -> "NumpyArray":  # type: ignore[override]
        return self._binary("Equal", self, other)

    def __ne__(self, other) -> "NumpyArray":  # type: ignore[override]
        eq = self._binary("Equal", self, other)
        return eq._unary("Not")

    def __hash__(self) -> int:
        # __eq__ is overloaded to return NumpyArray, so we must provide __hash__.
        return hash(id(self))

    # ------------------------------------------------------------------
    # Logical / bitwise
    # ------------------------------------------------------------------

    def __and__(self, other) -> "NumpyArray":
        return self._binary("And", self, other)

    def __rand__(self, other) -> "NumpyArray":
        return self._binary("And", other, self)

    def __or__(self, other) -> "NumpyArray":
        return self._binary("Or", self, other)

    def __ror__(self, other) -> "NumpyArray":
        return self._binary("Or", other, self)

    def __xor__(self, other) -> "NumpyArray":
        return self._binary("Xor", self, other)

    def __rxor__(self, other) -> "NumpyArray":
        return self._binary("Xor", other, self)

    def __invert__(self) -> "NumpyArray":
        return self._unary("Not")

    # ------------------------------------------------------------------
    # Matrix multiplication
    # ------------------------------------------------------------------

    def __matmul__(self, other) -> "NumpyArray":
        return self._binary("MatMul", self, other)

    def __rmatmul__(self, other) -> "NumpyArray":
        return self._binary("MatMul", other, self)

    # ------------------------------------------------------------------
    # Shape-related properties and methods
    # ------------------------------------------------------------------

    @property
    def T(self) -> "NumpyArray":
        """Transpose (reverses all axes)."""
        return self._transpose_no_perm(self._name)

    def _transpose_no_perm(self, input_name: str) -> "NumpyArray":
        """Emit a Transpose node, inferring ``perm`` from the known rank if possible."""
        g = self._g
        if g.has_rank(input_name):
            ndim = g.get_rank(input_name)
            perm = list(range(ndim - 1, -1, -1))
            res = g.op.Transpose(input_name, perm=perm, name=g.unique_name("transpose"))
        else:
            res = g.op.Transpose(input_name, name=g.unique_name("transpose"))
        return self._new(res)

    def transpose(self, *axes) -> "NumpyArray":
        """Transpose with optional *axes* permutation."""
        if not axes:
            return self._transpose_no_perm(self._name)
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = tuple(axes[0])
        res = self._g.op.Transpose(
            self._name, perm=list(axes), name=self._g.unique_name("transpose")
        )
        return self._new(res)

    def reshape(self, *shape) -> "NumpyArray":
        """Reshape the tensor."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape_arr = np.array(list(shape), dtype=np.int64)
        res = self._g.op.Reshape(self._name, shape_arr, name=self._g.unique_name("reshape"))
        return self._new(res)

    def flatten(self) -> "NumpyArray":
        """Flatten to a 1-D tensor."""
        res = self._g.op.Reshape(
            self._name, np.array([-1], dtype=np.int64), name=self._g.unique_name("flatten")
        )
        return self._new(res)

    def squeeze(self, axis=None) -> "NumpyArray":
        """Remove size-1 dimensions."""
        if axis is None:
            res = self._g.op.Squeeze(self._name, name=self._g.unique_name("squeeze"))
        else:
            axes_arr = np.array([axis] if isinstance(axis, int) else list(axis), dtype=np.int64)
            res = self._g.op.Squeeze(self._name, axes_arr, name=self._g.unique_name("squeeze"))
        return self._new(res)

    def expand_dims(self, axis) -> "NumpyArray":
        """Add a size-1 dimension (numpy ``expand_dims``)."""
        axes_arr = np.array([axis] if isinstance(axis, int) else list(axis), dtype=np.int64)
        res = self._g.op.Unsqueeze(self._name, axes_arr, name=self._g.unique_name("unsqueeze"))
        return self._new(res)

    # ------------------------------------------------------------------
    # Type casting
    # ------------------------------------------------------------------

    def astype(self, dtype) -> "NumpyArray":
        """Cast to *dtype*."""
        from ..helpers.onnx_helper import np_dtype_to_tensor_dtype

        np_dtype = np.dtype(dtype)
        to_type = np_dtype_to_tensor_dtype(np_dtype)
        res = self._g.op.Cast(self._name, to=to_type, name=self._g.unique_name("cast"))
        return NumpyArray(res, self._g, dtype=np_dtype)

    # ------------------------------------------------------------------
    # Reduction methods
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims: bool = False) -> "NumpyArray":
        """Sum elements along *axis*."""
        return self._reduce("ReduceSum", axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False) -> "NumpyArray":
        """Mean of elements along *axis*."""
        return self._reduce("ReduceMean", axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims: bool = False) -> "NumpyArray":
        """Maximum along *axis*."""
        return self._reduce("ReduceMax", axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims: bool = False) -> "NumpyArray":
        """Minimum along *axis*."""
        return self._reduce("ReduceMin", axis=axis, keepdims=keepdims)

    def prod(self, axis=None, keepdims: bool = False) -> "NumpyArray":
        """Product of elements along *axis*."""
        return self._reduce("ReduceProd", axis=axis, keepdims=keepdims)

    # ------------------------------------------------------------------
    # Clipping
    # ------------------------------------------------------------------

    def clip(self, a_min=None, a_max=None) -> "NumpyArray":
        """Clip values to ``[a_min, a_max]``."""
        ref_dtype = self._dtype if self._dtype is not None else np.float32
        clip_inputs: list = [self._name]
        if a_min is None:
            clip_inputs.append(None)
        else:
            clip_inputs.append(np.array(a_min, dtype=ref_dtype))
        if a_max is not None:
            clip_inputs.append(np.array(a_max, dtype=ref_dtype))
        res = self._g.op.Clip(*clip_inputs, name=self._g.unique_name("clip"))
        return self._new(res)

    # ------------------------------------------------------------------
    # Numpy dispatch protocols
    # ------------------------------------------------------------------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy ufuncs by delegating to ONNX operators."""
        if method != "__call__":
            return NotImplemented

        if ufunc not in _UFUNC_TO_ONNX:
            return NotImplemented

        # Find graph builder and reference dtype from first NumpyArray input.
        g = _get_g(*inputs)
        if g is None:
            return NotImplemented
        ref_dtype = next(
            (
                inp._dtype
                for inp in inputs
                if isinstance(inp, NumpyArray) and inp._dtype is not None
            ),
            None,
        )

        # Convert all inputs to Opset-compatible arguments.
        onnx_args = [_to_onnx_arg(inp, g, ref_dtype) for inp in inputs]

        mapping = _UFUNC_TO_ONNX[ufunc]

        # ---- special cases ----
        if mapping == "not_equal":
            eq = g.op.Equal(*onnx_args, name=g.unique_name("eq"))
            res = g.op.Not(eq, name=g.unique_name("not_equal"))
            return NumpyArray(res, g)

        if mapping == "floor_divide":
            div = g.op.Div(*onnx_args, name=g.unique_name("div"))
            res = g.op.Floor(div, name=g.unique_name("floor_divide"))
            return NumpyArray(res, g)

        if mapping == "maximum":
            # ONNX Max accepts variadic inputs
            res = g.op.Max(*onnx_args, name=g.unique_name("maximum"))
            return NumpyArray(res, g)

        if mapping == "minimum":
            res = g.op.Min(*onnx_args, name=g.unique_name("minimum"))
            return NumpyArray(res, g)

        if mapping == "log1p":
            one = np.array(1, dtype=ref_dtype if ref_dtype is not None else np.float32)
            added = g.op.Add(onnx_args[0], one, name=g.unique_name("add"))
            res = g.op.Log(added, name=g.unique_name("log1p"))
            return NumpyArray(res, g)

        if mapping == "expm1":
            exp = g.op.Exp(onnx_args[0], name=g.unique_name("exp"))
            one = np.array(1, dtype=ref_dtype if ref_dtype is not None else np.float32)
            res = g.op.Sub(exp, one, name=g.unique_name("expm1"))
            return NumpyArray(res, g)

        # ---- standard case ----
        onnx_op, _ = mapping
        op_method = getattr(g.op, onnx_op, None)
        if op_method is None:
            return NotImplemented

        res = op_method(*onnx_args, name=g.unique_name(onnx_op.lower()))
        return NumpyArray(res, g)

    def __array_function__(self, func, types, args, kwargs):
        """Handle numpy array functions via the ``__array_function__`` protocol."""
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"NumpyArray(name={self._name!r}, dtype={self._dtype!r}, shape={self._shape!r})"


# ---------------------------------------------------------------------------
# __array_function__ implementations
# ---------------------------------------------------------------------------


@_implements(np.reshape)
def _reshape(a, newshape, order="C"):
    """ONNX: Reshape"""
    if order != "C":
        raise NotImplementedError(
            f"reshape with order={order!r} is not supported for ONNX tracing."
        )
    shape_arr = np.array(list(newshape), dtype=np.int64)
    res = a._g.op.Reshape(a._name, shape_arr, name=a._g.unique_name("reshape"))
    return NumpyArray(res, a._g)


@_implements(np.transpose)
def _transpose(a, axes=None):
    """ONNX: Transpose"""
    if axes is None:
        return a._transpose_no_perm(a._name)
    res = a._g.op.Transpose(a._name, perm=list(axes), name=a._g.unique_name("transpose"))
    return NumpyArray(res, a._g)


@_implements(np.sum)
def _sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None):
    """ONNX: ReduceSum"""
    if initial is not None or where is not None or out is not None:
        return NotImplemented
    return a._reduce("ReduceSum", axis=axis, keepdims=keepdims)


@_implements(np.mean)
def _mean(a, axis=None, dtype=None, out=None, keepdims=False, where=None):
    """ONNX: ReduceMean"""
    if where is not None or out is not None:
        return NotImplemented
    return a._reduce("ReduceMean", axis=axis, keepdims=keepdims)


@_implements(np.amax)
def _amax(a, axis=None, out=None, keepdims=False, initial=None, where=None):
    """ONNX: ReduceMax"""
    if initial is not None or where is not None or out is not None:
        return NotImplemented
    return a._reduce("ReduceMax", axis=axis, keepdims=keepdims)


@_implements(np.amin)
def _amin(a, axis=None, out=None, keepdims=False, initial=None, where=None):
    """ONNX: ReduceMin"""
    if initial is not None or where is not None or out is not None:
        return NotImplemented
    return a._reduce("ReduceMin", axis=axis, keepdims=keepdims)


# In NumPy, np.max and np.min are aliases of np.amax/np.amin.
# Register them too in case the user writes np.max(arr, axis=…).
_HANDLED_FUNCTIONS[np.max] = _amax  # type: ignore[assignment]
_HANDLED_FUNCTIONS[np.min] = _amin  # type: ignore[assignment]


@_implements(np.prod)
def _prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None):
    """ONNX: ReduceProd"""
    if initial is not None or where is not None or out is not None:
        return NotImplemented
    return a._reduce("ReduceProd", axis=axis, keepdims=keepdims)


@_implements(np.clip)
def _clip(a, a_min=None, a_max=None, out=None, **kwargs):
    """ONNX: Clip"""
    if out is not None:
        return NotImplemented
    # numpy 2.x introduced min/max aliases; support both naming styles.
    if a_min is None:
        a_min = kwargs.pop("min", None)
    if a_max is None:
        a_max = kwargs.pop("max", None)
    return a.clip(a_min=a_min, a_max=a_max)


@_implements(np.where)
def _where(condition, x=None, y=None):
    """ONNX: Where"""
    if x is None or y is None:
        return NotImplemented
    g = _get_g(condition, x, y)
    if g is None:
        return NotImplemented
    ref_dtype = next(
        (v._dtype for v in (x, y) if isinstance(v, NumpyArray) and v._dtype is not None), None
    )
    cond_arg = _to_onnx_arg(condition, g)
    x_arg = _to_onnx_arg(x, g, ref_dtype)
    y_arg = _to_onnx_arg(y, g, ref_dtype)
    res = g.op.Where(cond_arg, x_arg, y_arg, name=g.unique_name("where"))
    return NumpyArray(res, g)


@_implements(np.concatenate)
def _concatenate(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    """ONNX: Concat"""
    if out is not None:
        return NotImplemented
    g = _get_g(*arrays)
    if g is None:
        return NotImplemented
    onnx_args = [_to_onnx_arg(a, g) for a in arrays]
    res = g.op.Concat(*onnx_args, axis=axis, name=g.unique_name("concat"))
    return NumpyArray(res, g)


@_implements(np.stack)
def _stack(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    """ONNX: Unsqueeze, Concat"""
    if out is not None:
        return NotImplemented
    g = _get_g(*arrays)
    if g is None:
        return NotImplemented
    axes_arr = np.array([axis], dtype=np.int64)
    unsqueezed = []
    for a in arrays:
        arg = _to_onnx_arg(a, g)
        uq = g.op.Unsqueeze(arg, axes_arr, name=g.unique_name("unsqueeze"))
        unsqueezed.append(uq)
    res = g.op.Concat(*unsqueezed, axis=axis, name=g.unique_name("stack"))
    return NumpyArray(res, g)


@_implements(np.expand_dims)
def _expand_dims(a, axis):
    """ONNX: Unsqueeze"""
    axes_arr = np.array([axis] if isinstance(axis, int) else list(axis), dtype=np.int64)
    res = a._g.op.Unsqueeze(a._name, axes_arr, name=a._g.unique_name("unsqueeze"))
    return NumpyArray(res, a._g)


@_implements(np.squeeze)
def _squeeze(a, axis=None):
    """ONNX: Squeeze"""
    if axis is None:
        res = a._g.op.Squeeze(a._name, name=a._g.unique_name("squeeze"))
    else:
        axes_arr = np.array([axis] if isinstance(axis, int) else list(axis), dtype=np.int64)
        res = a._g.op.Squeeze(a._name, axes_arr, name=a._g.unique_name("squeeze"))
    return NumpyArray(res, a._g)


@_implements(np.matmul)
def _matmul(a, b, out=None, **kwargs):
    """ONNX: MatMul"""
    if out is not None:
        return NotImplemented
    g = _get_g(a, b)
    if g is None:
        return NotImplemented
    ref_dtype = next(
        (v._dtype for v in (a, b) if isinstance(v, NumpyArray) and v._dtype is not None), None
    )
    res = g.op.MatMul(
        _to_onnx_arg(a, g, ref_dtype), _to_onnx_arg(b, g, ref_dtype), name=g.unique_name("matmul")
    )
    return NumpyArray(res, g)


@_implements(np.dot)
def _dot(a, b, out=None):
    """ONNX: MatMul"""
    if out is not None:
        return NotImplemented
    g = _get_g(a, b)
    if g is None:
        return NotImplemented
    ref_dtype = next(
        (v._dtype for v in (a, b) if isinstance(v, NumpyArray) and v._dtype is not None), None
    )
    res = g.op.MatMul(
        _to_onnx_arg(a, g, ref_dtype), _to_onnx_arg(b, g, ref_dtype), name=g.unique_name("dot")
    )
    return NumpyArray(res, g)


@_implements(np.abs)
def _abs(x, out=None, **kwargs):
    """ONNX: Abs"""
    if out is not None:
        return NotImplemented
    res = x._g.op.Abs(x._name, name=x._g.unique_name("abs"))
    return NumpyArray(res, x._g)


@_implements(np.sqrt)
def _sqrt(x, out=None, **kwargs):
    """ONNX: Sqrt"""
    if out is not None:
        return NotImplemented
    res = x._g.op.Sqrt(x._name, name=x._g.unique_name("sqrt"))
    return NumpyArray(res, x._g)


@_implements(np.exp)
def _exp(x, out=None, **kwargs):
    """ONNX: Exp"""
    if out is not None:
        return NotImplemented
    res = x._g.op.Exp(x._name, name=x._g.unique_name("exp"))
    return NumpyArray(res, x._g)


@_implements(np.log)
def _log(x, out=None, **kwargs):
    """ONNX: Log"""
    if out is not None:
        return NotImplemented
    res = x._g.op.Log(x._name, name=x._g.unique_name("log"))
    return NumpyArray(res, x._g)


@_implements(np.log1p)
def _log1p(x, out=None, **kwargs):
    """ONNX: Add, Log"""
    if out is not None:
        return NotImplemented
    # log1p(x) = log(1 + x)
    ref_dtype = x._dtype if x._dtype is not None else np.float32
    one = np.array(1, dtype=ref_dtype)
    added = x._g.op.Add(x._name, one, name=x._g.unique_name("add"))
    res = x._g.op.Log(added, name=x._g.unique_name("log1p"))
    return NumpyArray(res, x._g)


@_implements(np.expm1)
def _expm1(x, out=None, **kwargs):
    """ONNX: Exp, Sub"""
    if out is not None:
        return NotImplemented
    # expm1(x) = exp(x) - 1
    ref_dtype = x._dtype if x._dtype is not None else np.float32
    one = np.array(1, dtype=ref_dtype)
    exp = x._g.op.Exp(x._name, name=x._g.unique_name("exp"))
    res = x._g.op.Sub(exp, one, name=x._g.unique_name("expm1"))
    return NumpyArray(res, x._g)
