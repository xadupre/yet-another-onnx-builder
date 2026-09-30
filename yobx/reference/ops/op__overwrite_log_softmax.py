"""Implements dtype-preserving LogSoftmax through native runtime callbacks."""

import numpy
from ._native_op import NativeOpKernel, evaluate_native_operator


def log_softmax(data, axis):
    """Computes stable log probabilities with float32 accumulation for small floats."""
    if not -data.ndim <= axis < data.ndim:
        raise numpy.exceptions.AxisError(axis, data.ndim)
    if not data.size:
        return data.copy()
    values = data.astype(numpy.float32) if data.dtype.itemsize < 4 else data
    shifted = values - numpy.max(values, axis=axis, keepdims=True)
    result = shifted - numpy.log(numpy.sum(numpy.exp(shifted), axis=axis, keepdims=True))
    return result.astype(data.dtype, copy=False)


class LogSoftmax_1(NativeOpKernel):
    """Computes legacy LogSoftmax after flattening dimensions from the selected axis."""

    def _run(self, data, axis=1):
        if not -data.ndim <= axis < data.ndim:
            raise numpy.exceptions.AxisError(axis, data.ndim)
        axis %= data.ndim
        matrix = data.reshape(
            int(numpy.prod(data.shape[:axis], dtype=numpy.int64)),
            int(numpy.prod(data.shape[axis:], dtype=numpy.int64)),
        )
        return (log_softmax(matrix, 1).reshape(data.shape),)


class LogSoftmax_13(NativeOpKernel):
    """Computes LogSoftmax along one axis without flattening the input."""

    def _run(self, data, axis=-1):
        return (log_softmax(data, axis),)


class SoftmaxCrossEntropyLoss(NativeOpKernel):
    """Widens low-precision scores for the native float32 loss kernel."""

    def _run(self, scores, labels, weights=None, reduction="mean", ignore_index=None):
        dtype = scores.dtype
        if dtype.name in {"float16", "bfloat16"}:
            scores = scores.astype(numpy.float32)
            if weights is not None:
                weights = weights.astype(numpy.float32)
        outputs = evaluate_native_operator(
            "SoftmaxCrossEntropyLoss",
            scores,
            labels,
            weights,
            output_count=2,
            reduction=reduction,
            ignore_index=ignore_index,
        )
        return tuple(value.astype(dtype, copy=False) for value in outputs)
