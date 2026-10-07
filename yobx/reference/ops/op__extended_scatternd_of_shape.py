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
        if maskedValue is None:
            raise ValueError("MaskedScatterNDOfShape requires maskedValue.")
        data = np.zeros(shape, dtype=updates.dtype)
        masked = np.any(indices == maskedValue, axis=-1)
        filtered_indices = indices[~masked]
        filtered_updates = updates[~masked]
        (y,) = evaluate_native_operator(
            "ScatterND", data, filtered_indices, filtered_updates, reduction=reduction
        )
        return (y,)
