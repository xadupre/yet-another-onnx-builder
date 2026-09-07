import numpy as np
from ._native_op import NativeOpKernel


class Greater(NativeOpKernel):
    """Compares broadcast inputs without narrowing float64 values."""

    def _run(self, a, b):
        return (np.greater(a, b),)
