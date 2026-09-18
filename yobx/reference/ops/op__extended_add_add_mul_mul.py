import numpy as np
from ._native_op import NativeOpKernel


class AddAdd(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, x, y, z):
        return (x + y + z,)


class MulMul(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, x, y, z):
        return (x * y * z,)


class AddMul(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, x, y, z, transposeMiddle=None):
        res = (x + y) * z
        if transposeMiddle:
            res = np.transpose(res, axes=[0, 2, 1, 3])
        return (res,)


class MulAdd(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, x, y, z, transposeMiddle=None):
        res = (x * y) + z
        if transposeMiddle:
            res = np.transpose(res, axes=[0, 2, 1, 3])
        return (res,)


class SubMul(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, x, y, z, negative=None):
        if negative:
            return ((y - x) * z,)
        return ((x - y) * z,)


class MulSub(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, x, y, z, negative=None):
        if negative:
            return (z - (x * y),)
        return ((x * y) - z,)


class AddSharedInput(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, x, y, z):
        return (x + y, x + z)


class MulSharedInput(NativeOpKernel):
    op_domain = "yaourt.ortops.fused_kernel.cuda"

    def _run(self, x, y, z):
        return (x * y, x * z)
