"""Lowers JAX jaxprs directly to the native onnx-light graph builder."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
from jax.extend.core import ClosedJaxpr, Literal
import numpy as np

from .. import DEFAULT_TARGET_OPSET
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder

_ELEMENTWISE = {
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "div": "Div",
    "max": "Max",
    "min": "Min",
    "pow": "Pow",
    "neg": "Neg",
    "abs": "Abs",
    "exp": "Exp",
    "log": "Log",
    "sqrt": "Sqrt",
    "tanh": "Tanh",
    "sin": "Sin",
    "cos": "Cos",
    "logistic": "Sigmoid",
}


def _shape(var):
    """Returns the concrete shape of a jaxpr variable."""
    return tuple(var.aval.shape)


def _dtype(var):
    """Returns the ONNX dtype of a jaxpr variable."""
    return np_dtype_to_tensor_dtype(np.dtype(var.aval.dtype))


def _constant(g, value):
    """Registers a scalar or array literal as an ONNX initializer."""
    return g.make_initializer(g.unique_name("constant"), np.asarray(value))


def _value(g, names, var):
    """Resolves a jaxpr variable or literal to a graph value."""
    if isinstance(var, Literal):
        return _constant(g, var.val)
    return names[var]


def _record(g, names, eqn, values):
    """Records the outputs of a lowered equation and their JAX annotations."""
    if isinstance(values, str):
        values = [values]
    if len(values) != len(eqn.outvars):
        raise ValueError(f"{eqn.primitive.name} produced an unexpected number of outputs.")
    for var, value in zip(eqn.outvars, values):
        names[var] = value
        if not g.has_type(value):
            g.set_type(value, _dtype(var))


def _lower_dot(g, lhs, rhs, eqn):
    """Lowers dot_general with arbitrary contracting and batch axes."""
    (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
    lc, rc, lb, rb = tuple(lc), tuple(rc), tuple(lb), tuple(rb)
    ls, rs = _shape(eqn.invars[0]), _shape(eqn.invars[1])
    lf = [i for i in range(len(ls)) if i not in lc and i not in lb]
    rf = [i for i in range(len(rs)) if i not in rc and i not in rb]
    if len(lc) != len(rc) or len(lb) != len(rb):
        raise NotImplementedError("dot_general has inconsistent dimension numbers.")
    if any(ls[i] != rs[j] for i, j in zip(lc, rc)) or any(ls[i] != rs[j] for i, j in zip(lb, rb)):
        raise ValueError("dot_general contracting and batch dimensions must agree.")
    lp = list(lb) + lf + list(lc)
    rp = list(rb) + list(rc) + rf
    original_lhs, original_rhs = lhs, rhs
    if lp != list(range(len(ls))):
        lhs = g.op.Transpose(lhs, perm=lp)
    if rp != list(range(len(rs))):
        rhs = g.op.Transpose(rhs, perm=rp)
    if len(lf) != 1 or len(rf) != 1 or len(lc) != 1:

        def dimension(value, axes):
            if not axes:
                return _constant(g, np.asarray([1], dtype=np.int64))
            gathered = g.op.Gather(g.op.Shape(value), np.asarray(axes, dtype=np.int64), axis=0)
            return g.op.ReduceProd(gathered, np.asarray([0], dtype=np.int64), keepdims=1)

        def dimensions(value, axes):
            return g.op.Gather(g.op.Shape(value), np.asarray(axes, dtype=np.int64), axis=0)

        left_shape = (
            g.op.Concat(
                dimensions(original_lhs, lb),
                dimension(original_lhs, lf),
                dimension(original_lhs, lc),
                axis=0,
            )
            if lb
            else g.op.Concat(dimension(original_lhs, lf), dimension(original_lhs, lc), axis=0)
        )
        right_shape = (
            g.op.Concat(
                dimensions(original_rhs, rb),
                dimension(original_rhs, rc),
                dimension(original_rhs, rf),
                axis=0,
            )
            if rb
            else g.op.Concat(dimension(original_rhs, rc), dimension(original_rhs, rf), axis=0)
        )
        lhs = g.op.Reshape(lhs, left_shape)
        rhs = g.op.Reshape(rhs, right_shape)
    result = g.op.MatMul(lhs, rhs)
    if len(lf) != 1 or len(rf) != 1 or len(lc) != 1:
        parts = [
            g.op.Gather(g.op.Shape(value), np.asarray(axes, dtype=np.int64), axis=0)
            for value, axes in ((original_lhs, list(lb) + lf), (original_rhs, rf))
            if axes
        ]
        result = g.op.Reshape(result, g.op.Concat(*parts, axis=0) if len(parts) > 1 else parts[0])
    return result


def _lower_broadcast(g, names, eqn):
    """Lowers broadcast_in_dim using insertion and expansion of axes."""
    var = eqn.invars[0]
    value = _value(g, names, var)
    dimensions = tuple(eqn.params["broadcast_dimensions"])
    shape = _shape(eqn.outvars[0])
    missing = [axis for axis in range(len(shape)) if axis not in dimensions]
    if missing:
        value = g.op.Unsqueeze(value, np.asarray(missing, dtype=np.int64))
    inserted = tuple(
        1 if axis in missing else _shape(var)[dimensions.index(axis)]
        for axis in range(len(shape))
    )
    if inserted != shape:
        # Shapes that depend on the input's dynamic axes come from Shape rather
        # than the sample dimensions used when tracing the jaxpr.
        pieces = []
        for axis, size in enumerate(shape):
            if axis in dimensions and _shape(var)[dimensions.index(axis)] == size:
                pieces.append(
                    g.op.Gather(
                        g.op.Shape(_value(g, names, var)),
                        np.asarray(dimensions.index(axis), dtype=np.int64),
                        axis=0,
                    )
                )
            else:
                pieces.append(_constant(g, np.asarray(size, dtype=np.int64)))
        target = g.op.Concat(
            *[g.op.Unsqueeze(piece, np.asarray([0], dtype=np.int64)) for piece in pieces], axis=0
        )
        value = g.op.Expand(value, target)
    return value


def _lower_jaxpr(g, closed, inputs):
    """Lowers a closed jaxpr and returns its graph output names."""
    jaxpr = closed.jaxpr
    if len(inputs) != len(jaxpr.invars):
        raise ValueError("Incorrect number of inputs for nested JAX jaxpr.")
    names = dict(zip(jaxpr.invars, inputs))
    for var, const in zip(jaxpr.constvars, closed.consts):
        names[var] = _constant(g, np.asarray(const))
    for eqn in jaxpr.eqns:
        op = eqn.primitive.name
        args = [_value(g, names, var) for var in eqn.invars]
        if op in _ELEMENTWISE:
            result = getattr(g.op, _ELEMENTWISE[op])(*args)
        elif op == "dot_general":
            result = _lower_dot(g, *args, eqn)
        elif op == "broadcast_in_dim":
            result = _lower_broadcast(g, names, eqn)
        elif op in ("reduce_sum", "reduce_max", "reduce_min"):
            kind = {
                "reduce_sum": "ReduceSum",
                "reduce_max": "ReduceMax",
                "reduce_min": "ReduceMin",
            }[op]
            result = getattr(g.op, kind)(
                args[0], np.asarray(eqn.params["axes"], dtype=np.int64), keepdims=0
            )
        elif op in ("stop_gradient", "convert_element_type"):
            result = (
                g.op.Identity(args[0])
                if op == "stop_gradient"
                else g.op.Cast(args[0], to=_dtype(eqn.outvars[0]))
            )
        elif op in ("reshape", "squeeze"):
            result = g.op.Reshape(args[0], np.asarray(_shape(eqn.outvars[0]), dtype=np.int64))
        elif op == "transpose":
            result = g.op.Transpose(args[0], perm=list(eqn.params["permutation"]))
        elif op in ("jit", "custom_jvp_call", "custom_vjp_call"):
            inner = (
                eqn.params.get("jaxpr")
                or eqn.params.get("call_jaxpr")
                or eqn.params.get("fun_jaxpr")
            )
            if inner is None:
                raise NotImplementedError(f"JAX primitive {op!r} has no nested jaxpr.")
            if not isinstance(inner, ClosedJaxpr):
                inner = ClosedJaxpr(inner, ())
            result = _lower_jaxpr(g, inner, args)
        else:
            raise NotImplementedError(
                f"JAX primitive {op!r} is not supported by the direct converter."
            )
        _record(g, names, eqn, result)
    return [_value(g, names, var) for var in jaxpr.outvars]


def to_onnx(
    model: Callable,
    args: Tuple[Any, ...],
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Tuple[Dict[int, str], ...]] = None,
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    builder_cls: type = GraphBuilder,
    verbose: int = 0,
    large_model: bool = False,
    external_threshold: int = 1024,
    filename: Optional[str] = None,
    return_optimize_report: bool = False,
):
    """Converts a JAX callable directly into an onnx-light export artifact.

    Args:
        model: JAX callable accepting positional array inputs.
        args: Concrete sample inputs used to trace the JAX computation.
        input_names: Optional ONNX input names.
        dynamic_shapes: Per-input mappings from axes to symbolic dimension names.
            By default, axis zero of each non-scalar input is named ``batch``.
        target_opset: ONNX opset version or domain-to-version mapping.
        builder_cls: Native graph builder class.
        verbose: Builder verbosity.
        large_model: Enables external tensor storage.
        external_threshold: Minimum number of elements stored externally.
        filename: Optional path to save the resulting artifact.
        return_optimize_report: Includes native optimization statistics.

    Returns:
        An export artifact containing an onnx-light model.
    """
    arrays = tuple(np.asarray(arg) for arg in args)
    if not arrays:
        raise ValueError("JAX conversion requires at least one input.")
    if input_names is None:
        input_names = ["X" if len(arrays) == 1 else f"X{i}" for i in range(len(arrays))]
    if len(input_names) != len(arrays) or len(set(input_names)) != len(input_names):
        raise ValueError("input_names must contain one distinct name for every input.")
    if dynamic_shapes is not None and len(dynamic_shapes) != len(arrays):
        raise ValueError("dynamic_shapes must contain one axis mapping for every input.")
    opsets = {"": target_opset} if isinstance(target_opset, int) else dict(target_opset)
    opsets.setdefault("", DEFAULT_TARGET_OPSET)
    g = builder_cls(opsets, verbose=verbose)
    for i, (name, array) in enumerate(zip(input_names, arrays)):
        shape = list(array.shape)
        axes = ({0: "batch"} if shape else {}) if dynamic_shapes is None else dynamic_shapes[i]
        for axis, dimension in axes.items():
            if axis < 0 or axis >= len(shape) or not dimension:
                raise ValueError(f"Invalid dynamic axis {axis!r} for input {name!r}.")
            shape[axis] = dimension
        g.make_tensor_input(name, np_dtype_to_tensor_dtype(array.dtype), tuple(shape))
    closed = jax.make_jaxpr(model)(*arrays)
    for name in _lower_jaxpr(g, closed, input_names):
        g.make_tensor_output(name, indexed=False, allow_untyped_output=True)
    artifact = g.to_onnx(
        large_model=large_model,
        external_threshold=external_threshold,
        return_optimize_report=return_optimize_report,
    )
    if filename:
        artifact.save(filename)
    return artifact
