"""Preserves floating-point precision and signs in native Where callbacks."""

import numpy
from ._native_op import NativeOpKernel


class Where(NativeOpKernel):
    def _run(self, condition, x, y):
        if condition.dtype != numpy.bool_:
            raise TypeError(f"Where expects a boolean condition, got {condition.dtype}.")
        if x.dtype != y.dtype:
            raise TypeError(f"Where expects matching data types, got {x.dtype} and {y.dtype}.")
        return (numpy.where(condition, x, y),)
