import numpy as np
from ._native_op import NativeOpKernel
from ._native_op import evaluate_native_operator


class ScatterNDOfShape(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, shape, indices, updates, reduction=None, strategy=None):
        data = np.zeros(shape, dtype=updates.dtype)
        (y,) = evaluate_native_operator("ScatterND", data, indices, updates, reduction=reduction)
        return (y,)


class MaskedScatterNDOfShape(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, shape, indices, updates, reduction=None, maskedValue=None):
        data = np.zeros(shape, dtype=updates.dtype)
        new_updates = np.where(indices == maskedValue, 0, updates)
        (y,) = evaluate_native_operator(
            "ScatterND", data, indices, new_updates, reduction=reduction
        )
        return (y,)
