import numpy as np
from ._native_op import NativeOpKernel


def arg_reduce(data, reducer, axis=0, keepdims=1, select_last_index=0):
    """Computes int64 reduction indices with ONNX axis and tie semantics."""
    values = np.flip(data, axis=axis) if select_last_index else data
    indices = np.asarray(reducer(values, axis=axis), dtype=np.int64)
    if select_last_index:
        indices = data.shape[axis] - 1 - indices
    if keepdims:
        indices = np.expand_dims(indices, axis=axis)
    return indices


class ArgMax(NativeOpKernel):
    """Finds maximum-value indices without narrowing the input dtype."""

    def _run(self, data, axis=0, keepdims=1, select_last_index=0):
        return (arg_reduce(data, np.argmax, axis, keepdims, select_last_index),)


class ArgMin(NativeOpKernel):
    """Finds minimum-value indices without narrowing the input dtype."""

    def _run(self, data, axis=0, keepdims=1, select_last_index=0):
        return (arg_reduce(data, np.argmin, axis, keepdims, select_last_index),)
