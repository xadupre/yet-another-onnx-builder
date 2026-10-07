import numpy as np
from ._native_op import NativeOpKernel


class Transpose2DCastFP16(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, X):
        return (X.T.astype(np.float16),)


class Transpose2DCastFP32(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, X):
        return (X.T.astype(np.float32),)
