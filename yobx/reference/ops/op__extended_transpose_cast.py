import numpy as np
from onnx.reference.op_run import OpRun


class Transpose2DCastFP16(OpRun):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, X):
        return (X.T.astype(np.float16),)


class Transpose2DCastFP32(OpRun):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, X):
        return (X.T.astype(np.float32),)
