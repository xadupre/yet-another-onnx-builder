"""Lowers JAX jaxprs directly to the native onnx-light graph builder."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
from jax.extend.core import ClosedJaxpr, Literal
import numpy as np

from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
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
    "sign": "Sign",
    "ceil": "Ceil",
    "floor": "Floor",
    "logistic": "Sigmoid",
    "gt": "Greater",
    "lt": "Less",
    "ge": "GreaterOrEqual",
    "le": "LessOrEqual",
    "eq": "Equal",
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
    output_dtype = _dtype(eqn.outvars[0])
    if g.get_type(lhs) != output_dtype:
        lhs = g.op.Cast(lhs, to=output_dtype)
    if g.get_type(rhs) != output_dtype:
        rhs = g.op.Cast(rhs, to=output_dtype)
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
        final_shape = (
            g.op.Concat(*parts, axis=0)
            if len(parts) > 1
            else parts[0] if parts else np.asarray([], dtype=np.int64)
        )
        result = g.op.Reshape(result, final_shape)
    return result


def _lower_broadcast(g, names, eqn, dynamic_sources):
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
    needs_dynamic_expansion = any(
        axis not in dimensions and size in dynamic_sources for axis, size in enumerate(shape)
    )
    if inserted != shape or needs_dynamic_expansion:
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
            elif axis not in dimensions and size in dynamic_sources:
                source_info = dynamic_sources[size]
                if source_info is None:
                    raise NotImplementedError(
                        f"Cannot resolve broadcast axis {axis} of size {size}: "
                        "dynamic and other dimensions share this sample size."
                    )
                source, source_axis, _ = source_info
                pieces.append(
                    g.op.Gather(
                        g.op.Shape(source), np.asarray(source_axis, dtype=np.int64), axis=0
                    )
                )
            else:
                pieces.append(_constant(g, np.asarray(size, dtype=np.int64)))
        target = g.op.Concat(
            *[g.op.Unsqueeze(piece, np.asarray([0], dtype=np.int64)) for piece in pieces], axis=0
        )
        value = g.op.Expand(value, target)
    return value


def _lower_reshape(g, value, eqn):
    """Keeps a traced reshape polymorphic when its input has a dynamic axis."""
    output_shape = list(_shape(eqn.outvars[0]))
    input_shape = g.get_shape(value)
    dynamic_axes = [axis for axis, size in enumerate(input_shape) if isinstance(size, str)]
    if not dynamic_axes:
        return g.op.Reshape(value, np.asarray(output_shape, dtype=np.int64))
    if len(dynamic_axes) != 1:
        raise NotImplementedError("Reshape with multiple dynamic input axes is not supported.")
    sample_size = _shape(eqn.invars[0])[dynamic_axes[0]]
    if sample_size == 0:
        raise NotImplementedError("Reshape cannot infer a zero-sized dynamic dimension.")
    candidates = [axis for axis, size in enumerate(output_shape) if size == sample_size]
    if not candidates:
        candidates = [axis for axis, size in enumerate(output_shape) if size % sample_size == 0]
    if not candidates:
        raise NotImplementedError("Cannot locate dynamic dimension in JAX reshape.")
    dynamic_axis = candidates[0]
    same_size = output_shape[dynamic_axis] == sample_size
    output_shape[dynamic_axis] = -1
    result = g.op.Reshape(value, np.asarray(output_shape, dtype=np.int64))
    symbolic = input_shape[dynamic_axes[0]] if same_size else g.unique_dimension_name("reshape")
    declared = list(_shape(eqn.outvars[0]))
    declared[dynamic_axis] = symbolic
    g.set_shape(result, tuple(declared))
    return result


def _lower_jaxpr(g, closed, inputs, dynamic_sources):
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
        elif op == "ne":
            result = g.op.Not(g.op.Equal(*args))
        elif op == "dot_general":
            if len(args) != 2:
                raise ValueError(f"dot_general expects two inputs, not {len(args)}.")
            result = _lower_dot(g, args[0], args[1], eqn)
        elif op == "broadcast_in_dim":
            result = _lower_broadcast(g, names, eqn, dynamic_sources)
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
            result = _lower_reshape(g, args[0], eqn)
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
            result = _lower_jaxpr(g, inner, args, dynamic_sources)
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
    builder_cls: Union[type, Callable] = GraphBuilder,
    verbose: int = 0,
    extra_converters: Optional[Dict[str, Callable]] = None,
    large_model: bool = False,
    external_threshold: int = 1024,
    filename: Optional[str] = None,
    return_optimize_report: bool = False,
) -> ExportArtifact:
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
        extra_converters: Reserved for signature compatibility. It must be ``None``.
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
    if (
        len(input_names) != len(arrays)
        or any(not isinstance(name, str) or not name for name in input_names)
        or len(set(input_names)) != len(input_names)
    ):
        raise ValueError("input_names must contain one distinct name for every input.")
    if dynamic_shapes is not None and len(dynamic_shapes) != len(arrays):
        raise ValueError("dynamic_shapes must contain one axis mapping for every input.")
    if extra_converters is not None:
        raise ValueError("extra_converters is reserved and must be None for JAX conversion.")
    opsets = {"": target_opset} if isinstance(target_opset, int) else dict(target_opset)
    opsets.setdefault("", DEFAULT_TARGET_OPSET)
    closed = jax.make_jaxpr(model)(*arrays)
    for array, var in zip(arrays, closed.jaxpr.invars):
        aval: Any = var.aval
        if array.dtype != np.dtype(aval.dtype):
            raise TypeError(
                f"JAX traced input dtype {aval.dtype} differs from sample dtype "
                f"{array.dtype}; enable the required JAX dtype before converting."
            )
    g = builder_cls(opsets, verbose=verbose)
    dynamic_sources: Dict[int, Optional[Tuple[str, int, str]]] = {}
    static_sizes: set[int] = set()
    for i, (name, array) in enumerate(zip(input_names, arrays)):
        shape = list(array.shape)
        axes = ({0: "batch"} if shape else {}) if dynamic_shapes is None else dynamic_shapes[i]
        static_sizes.update(size for axis, size in enumerate(array.shape) if axis not in axes)
        for axis, dimension in axes.items():
            if (
                not isinstance(axis, int)
                or axis < 0
                or axis >= len(shape)
                or not isinstance(dimension, str)
                or not dimension
            ):
                raise ValueError(f"Invalid dynamic axis {axis!r} for input {name!r}.")
            shape[axis] = dimension
            sample_size = array.shape[axis]
            if sample_size not in dynamic_sources:
                dynamic_sources[sample_size] = (name, axis, dimension)
            else:
                source_info = dynamic_sources[sample_size]
                if source_info is not None and source_info[2] != dimension:
                    dynamic_sources[sample_size] = None
        g.make_tensor_input(name, np_dtype_to_tensor_dtype(array.dtype), tuple(shape))
    for size in static_sizes.intersection(dynamic_sources):
        dynamic_sources[size] = None
    outputs = set()
    for name in _lower_jaxpr(g, closed, input_names, dynamic_sources):
        if name in outputs:
            name = g.op.Identity(name)
        g.make_tensor_output(name, indexed=False, allow_untyped_output=True)
        outputs.add(name)
    artifact = g.to_onnx(
        large_model=large_model,
        external_threshold=external_threshold,
        return_optimize_report=return_optimize_report,
    )
    if filename:
        artifact.save(filename)
    return artifact
