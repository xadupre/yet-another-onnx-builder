from ._native_op import NativeOpKernel
from ._native_op import evaluate_native_operator


class SkipLayerNormalization(NativeOpKernel):
    op_domain = "com.microsoft"

    def _run(self, x, skip, gamma=None, beta=None, bias=None, epsilon=None):
        add = x + skip
        if bias is not None:
            add = add + bias
        res = evaluate_native_operator(
            "LayerNormalization", add, gamma, beta, axis=-1, epsilon=epsilon, output_count=3
        )
        return (*res, add)
