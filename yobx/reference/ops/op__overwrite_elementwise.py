"""Preserves standard operator dtypes through explicit native callbacks."""

import numpy
import ml_dtypes
from ._native_op import NativeOpKernel


class Pow(NativeOpKernel):
    """Computes broadcast powers while preserving the base tensor's dtype."""

    def _run(self, x, y):
        return (numpy.power(x, y).astype(x.dtype, copy=False),)


class Min(NativeOpKernel):
    """Computes a variadic broadcast minimum with NaN propagation."""

    operation = staticmethod(numpy.minimum)

    def _run(self, *inputs):
        result = inputs[0]
        for value in inputs[1:]:
            result = self.operation(result, value)
        return (result,)


class Max(Min):
    """Computes a variadic broadcast maximum with NaN propagation."""

    operation = staticmethod(numpy.maximum)


class IsInf(NativeOpKernel):
    """Detects selected signs of infinity without changing the input precision."""

    def _run(self, x, detect_negative=1, detect_positive=1):
        result = numpy.isinf(x)
        if not detect_negative:
            result &= x > 0
        if not detect_positive:
            result &= x < 0
        return (result,)


class NonZero(NativeOpKernel):
    """Returns row-major nonzero coordinates, including ONNX scalar shapes."""

    def _run(self, x):
        if x.ndim == 0:
            return (numpy.empty((0, int(numpy.count_nonzero(x))), dtype=numpy.int64),)
        return (numpy.asarray(numpy.nonzero(x), dtype=numpy.int64),)


class Clip_11(NativeOpKernel):
    """Clips tensors using optional scalar bounds and dtype-specific defaults."""

    def _run(self, x, min=None, max=None):
        if min is None or max is None:
            limits = (
                ml_dtypes.finfo(x.dtype)
                if x.dtype.name == "bfloat16"
                else numpy.iinfo(x.dtype) if x.dtype.kind in "iu" else numpy.finfo(x.dtype)
            )
            min = limits.min if min is None else min
            max = limits.max if max is None else max
        return (
            numpy.minimum(
                numpy.maximum(x, numpy.asarray(min, dtype=x.dtype)),
                numpy.asarray(max, dtype=x.dtype),
            ),
        )


class Clip_6(Clip_11):
    """Clips tensors using the legacy float attribute bounds."""

    def _run(self, x, min=-3.4028234663852886e38, max=3.4028234663852886e38):
        return super()._run(x, min=min, max=max)


class HardSigmoid(NativeOpKernel):
    """Computes clipped affine activations with widened low-precision arithmetic."""

    def _run(self, x, alpha=0.2, beta=0.5):
        data = x.astype(numpy.float32) if x.dtype.name in {"float16", "bfloat16"} else x
        return (numpy.clip(data * alpha + beta, 0, 1).astype(x.dtype, copy=False),)
