from ._native_op import NativeOpKernel


class SimplifiedLayerNormalization(NativeOpKernel):
    def _run(self, x, scale, bias=None, axis=None, epsilon=None, stash_type=None):
        xm = (x**2).mean(axis=axis, keepdims=1) + epsilon
        xq = xm ** (-0.5)
        return (x * xq, xq)
