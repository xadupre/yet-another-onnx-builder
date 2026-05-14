import numpy as np
from onnx.reference.op_run import OpRun
from .op_fast_gelu import FastGelu


class GemmFastGelu(OpRun):
    """Implements the ``com.microsoft.GemmFastGelu`` operator.

    Computes ``FastGelu(A @ B + bias)`` where bias is optional.
    """

    op_domain = "com.microsoft"

    def _run(self, A, B, bias=None):
        y = np.matmul(A, B)
        if bias is not None:
            y = y + bias
        return (FastGelu._fast_gelu_core(y).astype(A.dtype),)
