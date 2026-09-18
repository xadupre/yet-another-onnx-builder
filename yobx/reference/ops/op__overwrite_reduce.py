"""Preserves reduction dtypes through the native custom-kernel interface."""

import numpy
from ._native_op import NativeOpKernel


def reduction_axes(axes):
    """Normalizes input or attribute axes, including the reduce-all default."""
    if axes is None:
        return None
    values = numpy.asarray(axes, dtype=numpy.int64).reshape(-1)
    return tuple(values.tolist()) if values.size else None


class ReduceMin(NativeOpKernel):
    """Computes minimum reductions, including float64 and empty tensors."""

    minimum = True

    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        axes = reduction_axes(axes)
        if axes is None and noop_with_empty_axes:
            return (data,)
        if data.dtype.kind in "iu":
            limits = numpy.iinfo(data.dtype)
            initial = limits.max if self.minimum else limits.min
        elif data.dtype.kind == "b":
            initial = self.minimum
        else:
            initial = numpy.inf if self.minimum else -numpy.inf
        operation = numpy.minimum if self.minimum else numpy.maximum
        return (operation.reduce(data, axis=axes, keepdims=bool(keepdims), initial=initial),)


class ReduceMax(ReduceMin):
    """Computes maximum reductions, including float64 and empty tensors."""

    minimum = False


class ReduceMean(NativeOpKernel):
    """Computes mean reductions without narrowing floating-point inputs."""

    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        axes = reduction_axes(axes)
        if axes is None and noop_with_empty_axes:
            return (data,)
        return (
            numpy.mean(data, axis=axes, keepdims=bool(keepdims)).astype(data.dtype, copy=False),
        )


class ReduceSum(NativeOpKernel):
    """Computes sums with exact integer and widened low-precision accumulation."""

    operation = staticmethod(numpy.sum)

    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        axes = reduction_axes(axes)
        if axes is None and noop_with_empty_axes:
            return (data,)
        dtype = numpy.float32 if data.dtype.name in {"float16", "bfloat16"} else data.dtype
        return (
            self.operation(data, axis=axes, keepdims=bool(keepdims), dtype=dtype).astype(
                data.dtype, copy=False
            ),
        )


class ReduceProd(ReduceSum):
    """Computes products without converting integer inputs to floating point."""

    operation = staticmethod(numpy.prod)
