"""Lowers JAX linear algebra primitives."""

import numpy as np

from ._registry import jax_dtype, jax_shape, register_lowering


def _lower_dot(state, eqn, args):
    """Lowers dot_general with arbitrary contracting and batch axes."""
    if len(args) != 2:
        raise ValueError(f"dot_general expects two inputs, not {len(args)}.")
    builder = state.builder
    lhs, rhs = args
    output_dtype = jax_dtype(eqn.outvars[0])
    if builder.get_type(lhs) != output_dtype:
        lhs = builder.op.Cast(lhs, to=output_dtype)
    if builder.get_type(rhs) != output_dtype:
        rhs = builder.op.Cast(rhs, to=output_dtype)
    (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
    lc, rc, lb, rb = tuple(lc), tuple(rc), tuple(lb), tuple(rb)
    ls, rs = jax_shape(eqn.invars[0]), jax_shape(eqn.invars[1])
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
        lhs = builder.op.Transpose(lhs, perm=lp)
    if rp != list(range(len(rs))):
        rhs = builder.op.Transpose(rhs, perm=rp)
    if len(lf) != 1 or len(rf) != 1 or len(lc) != 1:

        def dimension(value, axes):
            if not axes:
                return state.constant(np.asarray([1], dtype=np.int64))
            gathered = builder.op.Gather(
                builder.op.Shape(value), np.asarray(axes, dtype=np.int64), axis=0
            )
            return builder.op.ReduceProd(gathered, np.asarray([0], dtype=np.int64), keepdims=1)

        def dimensions(value, axes):
            return builder.op.Gather(
                builder.op.Shape(value), np.asarray(axes, dtype=np.int64), axis=0
            )

        left_shape = (
            builder.op.Concat(
                dimensions(original_lhs, lb),
                dimension(original_lhs, lf),
                dimension(original_lhs, lc),
                axis=0,
            )
            if lb
            else builder.op.Concat(
                dimension(original_lhs, lf), dimension(original_lhs, lc), axis=0
            )
        )
        right_shape = (
            builder.op.Concat(
                dimensions(original_rhs, rb),
                dimension(original_rhs, rc),
                dimension(original_rhs, rf),
                axis=0,
            )
            if rb
            else builder.op.Concat(
                dimension(original_rhs, rc), dimension(original_rhs, rf), axis=0
            )
        )
        lhs = builder.op.Reshape(lhs, left_shape)
        rhs = builder.op.Reshape(rhs, right_shape)
    result = builder.op.MatMul(lhs, rhs)
    if len(lf) != 1 or len(rf) != 1 or len(lc) != 1:
        parts = [
            builder.op.Gather(builder.op.Shape(value), np.asarray(axes, dtype=np.int64), axis=0)
            for value, axes in ((original_lhs, list(lb) + lf), (original_rhs, rf))
            if axes
        ]
        final_shape = (
            builder.op.Concat(*parts, axis=0)
            if len(parts) > 1
            else parts[0] if parts else np.asarray([], dtype=np.int64)
        )
        result = builder.op.Reshape(result, final_shape)
    return result


register_lowering("dot_general", _lower_dot)
