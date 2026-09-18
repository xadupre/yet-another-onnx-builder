from ._native_op import NativeOpKernel


class MemcpyFromHost(NativeOpKernel):
    def _run(self, x):
        return (x,)


class MemcpyToHost(NativeOpKernel):
    def _run(self, x):
        return (x,)
