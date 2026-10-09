"""Lowers JAX tensor-shape and reduction primitives."""

import numpy as np

from ._registry import jax_shape, register_lowering


def _lower_broadcast(state, eqn, args):
    """Lowers broadcast_in_dim using insertion and expansion of axes."""
    builder = state.builder
    var = eqn.invars[0]
    value = args[0]
    dimensions = tuple(eqn.params["broadcast_dimensions"])
    shape = jax_shape(eqn.outvars[0])
    missing = [axis for axis in range(len(shape)) if axis not in dimensions]
    if missing:
        value = builder.op.Unsqueeze(value, np.asarray(missing, dtype=np.int64))
    inserted = tuple(
        1 if axis in missing else jax_shape(var)[dimensions.index(axis)]
        for axis in range(len(shape))
    )
    needs_dynamic_expansion = any(
        axis not in dimensions and size in state.dynamic_sources
        for axis, size in enumerate(shape)
    )
    if inserted != shape or needs_dynamic_expansion:
        pieces = []
        for axis, size in enumerate(shape):
            if axis in dimensions and jax_shape(var)[dimensions.index(axis)] == size:
                pieces.append(
                    builder.op.Gather(
                        builder.op.Shape(args[0]),
                        np.asarray(dimensions.index(axis), dtype=np.int64),
                        axis=0,
                    )
                )
            elif axis not in dimensions and size in state.dynamic_sources:
                source_info = state.dynamic_sources[size]
                if source_info is None:
                    raise NotImplementedError(
                        f"Cannot resolve broadcast axis {axis} of size {size}: "
                        "dynamic and other dimensions share this sample size."
                    )
                source, source_axis, _ = source_info
                pieces.append(
                    builder.op.Gather(
                        builder.op.Shape(source), np.asarray(source_axis, dtype=np.int64), axis=0
                    )
                )
            else:
                pieces.append(state.constant(np.asarray(size, dtype=np.int64)))
        target = builder.op.Concat(
            *[builder.op.Unsqueeze(piece, np.asarray([0], dtype=np.int64)) for piece in pieces],
            axis=0,
        )
        value = builder.op.Expand(value, target)
    return value


def _lower_reshape(state, eqn, args):
    """Keeps a traced reshape polymorphic when its input has a dynamic axis."""
    builder = state.builder
    value = args[0]
    output_shape = list(jax_shape(eqn.outvars[0]))
    input_shape = builder.get_shape(value)
    dynamic_axes = [axis for axis, size in enumerate(input_shape) if isinstance(size, str)]
    if not dynamic_axes:
        return builder.op.Reshape(value, np.asarray(output_shape, dtype=np.int64))
    if len(dynamic_axes) != 1:
        raise NotImplementedError("Reshape with multiple dynamic input axes is not supported.")
    sample_size = jax_shape(eqn.invars[0])[dynamic_axes[0]]
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
    result = builder.op.Reshape(value, np.asarray(output_shape, dtype=np.int64))
    symbolic = (
        input_shape[dynamic_axes[0]] if same_size else builder.unique_dimension_name("reshape")
    )
    declared = list(jax_shape(eqn.outvars[0]))
    declared[dynamic_axis] = symbolic
    builder.set_shape(result, tuple(declared))
    return result


def _lower_reduction(state, eqn, args):
    """Lowers a JAX reduction primitive."""
    op_type = {"reduce_sum": "ReduceSum", "reduce_max": "ReduceMax", "reduce_min": "ReduceMin"}[
        eqn.primitive.name
    ]
    return getattr(state.builder.op, op_type)(
        args[0], np.asarray(eqn.params["axes"], dtype=np.int64), keepdims=0
    )


def _lower_transpose(state, eqn, args):
    """Lowers a JAX transpose primitive."""
    return state.builder.op.Transpose(args[0], perm=list(eqn.params["permutation"]))


register_lowering("broadcast_in_dim", _lower_broadcast)
register_lowering("reshape", _lower_reshape)
register_lowering("squeeze", _lower_reshape)
register_lowering("reduce_sum", _lower_reduction)
register_lowering("reduce_max", _lower_reduction)
register_lowering("reduce_min", _lower_reduction)
register_lowering("transpose", _lower_transpose)
