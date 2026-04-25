import numpy as np
from onnx.reference.op_run import OpRun
from .op_fast_gelu import _fast_gelu_core


class GemmFastGelu(OpRun):
    op_domain = "com.microsoft"

    def _run(self, A, B, bias=None):
        y = np.matmul(A, B)
        if bias is not None:
            y = y + bias
        return (_fast_gelu_core(y).astype(A.dtype),)
