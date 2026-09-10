from ._native_op import NativeOpKernel


class NegXplus1(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, X):
        return ((1 - X).astype(X.dtype),)
