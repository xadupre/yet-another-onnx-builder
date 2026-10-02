"""Preserves string tensors through the native custom-kernel interface."""

import numpy
from ._native_op import NativeOpKernel


class Compress(NativeOpKernel):
    """Selects elements of numeric or string tensors along the requested axis."""

    def _run(self, data, condition, axis=None):
        return (numpy.compress(condition, data, axis=axis),)
