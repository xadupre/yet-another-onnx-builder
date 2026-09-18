import numpy as np
from ._native_op import NativeOpKernel


def sigmoid(x):  # type: ignore
    if x > 0:
        return 1 / (1 + np.exp(-x))
    return np.exp(x) / (1 + np.exp(x))


class MulSigmoid(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def __init__(self, onnx_node, run_params):  # type: ignore
        NativeOpKernel.__init__(self, onnx_node, run_params)
        self.vf = np.vectorize(sigmoid)

    def _run(self, X):
        if len(X.shape) == 0:
            return ((X * sigmoid(X)).astype(X.dtype),)
        if X.size == 0:
            return (X,)
        return ((X * self.vf(X)).astype(X.dtype),)
