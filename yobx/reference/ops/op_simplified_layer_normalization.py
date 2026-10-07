import numpy as np
from onnx_light.onnx import helper

from ._native_op import NativeOpKernel


class SimplifiedLayerNormalization(NativeOpKernel):
    """Implements ``com.microsoft.SimplifiedLayerNormalization``."""

    op_domain = "com.microsoft"

    def _run(self, x, scale, axis=-1, epsilon=1.0e-5, stash_type=None):
        if axis < 0:
            axis += x.ndim
        axes = tuple(range(axis, x.ndim))
        compute_dtype = (
            np.dtype(np.float32)
            if stash_type is None
            else np.dtype(helper.tensor_dtype_to_np_dtype(stash_type))
        )
        squared = np.square(x.astype(compute_dtype, copy=False))
        inv_std_var = np.reciprocal(np.sqrt(squared.mean(axis=axes, keepdims=True) + epsilon))
        return ((x * inv_std_var * scale).astype(scale.dtype), inv_std_var)
