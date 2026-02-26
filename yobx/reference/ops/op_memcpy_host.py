from onnx.reference.op_run import OpRun


class MemcpyFromHost(OpRun):
    op_domain = "com.microsoft"

    def _run(self, x):
        return (x,)


class MemcpyToHost(OpRun):
    op_domain = "com.microsoft"

    def _run(self, x):
        return (x,)
