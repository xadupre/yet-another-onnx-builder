"""
StableHLO → ONNX op mappings for ``XlaCallModule`` conversion.

When a JAX function is lowered through ``jax2tf``, TensorFlow wraps the
computation in an ``XlaCallModule`` op whose payload is a StableHLO MLIR
module.  :func:`get_jax_cvt` is the single look-up point used by the
``XlaCallModule`` converter to find the appropriate ONNX emission callable
for each ``stablehlo.*`` op encountered in the parsed MLIR.

Direct mappings (:data:`_MAPPING_JAX_ONNX`)
--------------------------------------------
StableHLO unary and binary ops that map 1-to-1 to a single ONNX op.

Composite mappings (:data:`_COMPOSITE_JAX_OPS`)
------------------------------------------------
StableHLO ops that require more than one ONNX node; implemented as small
factory functions that close over the :class:`GraphBuilderExtendedProtocol`
instance.
"""

from typing import Any, Union

from ...typing import GraphBuilderExtendedProtocol

# ---------------------------------------------------------------------------
# Direct 1-to-1 mappings
# ---------------------------------------------------------------------------

# Mapping from StableHLO op names (after stripping the ``stablehlo.``
# prefix) to their direct ONNX-op-name equivalents.  Only truly 1-to-1 ops
# belong here; composite ops are handled via :data:`_COMPOSITE_JAX_OPS`.
_MAPPING_JAX_ONNX: dict = {
    # -----------------------------------------------------------------------
    # Unary ops
    # -----------------------------------------------------------------------
    # Magnitude / rounding
    "abs": "Abs",
    "ceil": "Ceil",
    "floor": "Floor",
    "negate": "Neg",
    "round_nearest_even": "Round",
    "sign": "Sign",
    # Exponential / logarithm
    "exponential": "Exp",
    "log": "Log",
    # Trigonometric
    "cosine": "Cos",
    "sine": "Sin",
    # Hyperbolic
    "tanh": "Tanh",
    # Activation
    "logistic": "Sigmoid",
    # Square-root
    "sqrt": "Sqrt",
    # Bitwise / logical unary
    "not": "Not",
    # -----------------------------------------------------------------------
    # Binary arithmetic ops
    # -----------------------------------------------------------------------
    "add": "Add",
    "subtract": "Sub",
    "multiply": "Mul",
    "divide": "Div",
    "maximum": "Max",
    "minimum": "Min",
    "power": "Pow",
    "remainder": "Mod",
    # -----------------------------------------------------------------------
    # Binary bitwise / logical ops
    # -----------------------------------------------------------------------
    "and": "And",
    "or": "Or",
    "xor": "Xor",
    # -----------------------------------------------------------------------
    # Selection (ternary but acts like a binary-class op)
    # -----------------------------------------------------------------------
    "select": "Where",
    # -----------------------------------------------------------------------
    # Comparison ops (direction encoded in op name by parse_mlir)
    # stablehlo.compare is rewritten as compare_<DIRECTION> during MLIR parsing
    # -----------------------------------------------------------------------
    "compare_EQ": "Equal",
    "compare_GT": "Greater",
    "compare_GE": "GreaterOrEqual",
    "compare_LT": "Less",
    "compare_LE": "LessOrEqual",
}


# ---------------------------------------------------------------------------
# Composite op factories
# ---------------------------------------------------------------------------


def _make_rsqrt(g: GraphBuilderExtendedProtocol):
    """Return a callable for ``stablehlo.rsqrt`` → ``Reciprocal(Sqrt(x))``."""

    def _rsqrt(*args, **kwargs):
        name = kwargs.pop("name", "rsqrt")
        outputs = kwargs.pop("outputs", None)
        (x,) = args
        sqrt = g.op.Sqrt(x, name=f"{name}_sqrt")
        kw = {"name": name}
        if outputs is not None:
            kw["outputs"] = outputs
        return g.op.Reciprocal(sqrt, **kw)

    return _rsqrt


def _make_log_plus_one(g: GraphBuilderExtendedProtocol):
    """Return a callable for ``stablehlo.log_plus_one`` → ``Log(Add(x, 1))``."""
    import numpy as np
    from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

    def _log1p(*args, **kwargs):
        name = kwargs.pop("name", "log_plus_one")
        outputs = kwargs.pop("outputs", None)
        (x,) = args
        try:
            dtype = tensor_dtype_to_np_dtype(g.get_type(x))
        except (AssertionError, KeyError):
            dtype = np.float32
        one = np.array(1, dtype=dtype)
        xp1 = g.op.Add(x, one, name=f"{name}_add")
        kw = {"name": name}
        if outputs is not None:
            kw["outputs"] = outputs
        return g.op.Log(xp1, **kw)

    return _log1p


def _make_exponential_minus_one(g: GraphBuilderExtendedProtocol):
    """Return a callable for ``stablehlo.exponential_minus_one`` → ``Sub(Exp(x), 1)``."""
    import numpy as np
    from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

    def _expm1(*args, **kwargs):
        name = kwargs.pop("name", "exponential_minus_one")
        outputs = kwargs.pop("outputs", None)
        (x,) = args
        try:
            dtype = tensor_dtype_to_np_dtype(g.get_type(x))
        except (AssertionError, KeyError):
            dtype = np.float32
        one = np.array(1, dtype=dtype)
        exp_x = g.op.Exp(x, name=f"{name}_exp")
        kw = {"name": name}
        if outputs is not None:
            kw["outputs"] = outputs
        return g.op.Sub(exp_x, one, **kw)

    return _expm1


def _make_compare_ne(g: GraphBuilderExtendedProtocol):
    """Return a callable for ``stablehlo.compare NE`` → ``Not(Equal(a, b))``."""

    def _ne(*args, **kwargs):
        name = kwargs.pop("name", "compare_ne")
        outputs = kwargs.pop("outputs", None)
        a, b = args
        eq = g.op.Equal(a, b, name=f"{name}_eq")
        kw = {"name": name}
        if outputs is not None:
            kw["outputs"] = outputs
        return g.op.Not(eq, **kw)

    return _ne


# Factory functions for composite ops that cannot be expressed as a single
# ONNX op.  Each factory receives the :class:`GraphBuilderExtendedProtocol`
# instance and returns a callable with the same signature as a simple
# ``g.op.<OpName>`` call.
_COMPOSITE_JAX_OPS: dict = {
    "rsqrt": _make_rsqrt,
    "log_plus_one": _make_log_plus_one,
    "exponential_minus_one": _make_exponential_minus_one,
    # compare_NE: stablehlo.compare NE → Not(Equal(a, b))
    "compare_NE": _make_compare_ne,
}


# ---------------------------------------------------------------------------
# Look-up function
# ---------------------------------------------------------------------------


def get_jax_cvt(assembly_code: Union[str, Any], g: GraphBuilderExtendedProtocol, jax_type: str):
    """Return an ONNX-emission callable for StableHLO *jax_type*.

    :param assembly_code: full MLIR text or ``ir.Module`` (used only in the
        error message).
    :param g: the active :class:`~yobx.typing.GraphBuilderExtendedProtocol`.
    :param jax_type: StableHLO op name with the ``stablehlo.`` prefix already
        stripped (e.g. ``"sine"``, ``"sqrt"``).
    :raises RuntimeError: if *jax_type* has no registered converter.
    """
    if jax_type in _MAPPING_JAX_ONNX:
        return lambda *args, **kwargs: getattr(g.op, _MAPPING_JAX_ONNX[jax_type])(*args, **kwargs)
    if jax_type in _COMPOSITE_JAX_OPS:
        return _COMPOSITE_JAX_OPS[jax_type](g)
    raise RuntimeError(
        f"Unable to handle jax operator {jax_type!r} in\n{assembly_code}{g.get_debug_msg()}"
    )
