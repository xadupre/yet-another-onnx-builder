import numpy as np
from onnx.reference.op_run import OpRun


class ToComplex(OpRun):
    op_domain = "ai.onnx.complex"

    def _run(self, x):
        assert x.shape[-1] in (1, 2), f"Unexpected shape {x.shape}, it should a tensor (..., 2)"
        if x.shape[-1] == 1:
            return (x[..., 0] + 0j,)
        return (x[..., 0] + 1j * x[..., 1],)


class ComplexModule(OpRun):
    op_domain = "ai.onnx.complex"

    def _run(self, x):
        assert x.dtype in (
            np.complex64,
            np.complex128,
        ), f"Unexpected type {x.dtype}, it should a complex tensor"
        return (np.abs(x),)


class ComplexMul(OpRun):
    """Implements ``com.microsoft.ComplexMul``.

    Computes element-wise complex multiplication of two tensors whose last
    dimension has size 2 (real, imaginary)::

        C[..., 0] = A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1]
        C[..., 1] = A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]
    """

    op_domain = "com.microsoft"

    def _run(self, A, B):
        assert A.shape[-1] == 2, f"Expected last dim 2, got {A.shape}"
        assert B.shape[-1] == 2, f"Expected last dim 2, got {B.shape}"
        a_r, a_i = A[..., 0], A[..., 1]
        b_r, b_i = B[..., 0], B[..., 1]
        c_r = a_r * b_r - a_i * b_i
        c_i = a_r * b_i + a_i * b_r
        return (np.stack([c_r, c_i], axis=-1),)


class ComplexMulConj(OpRun):
    """Implements ``com.microsoft.ComplexMulConj``.

    Computes element-wise complex multiplication of ``A`` with the conjugate of
    ``B``.  Both tensors must have a last dimension of size 2 (real, imaginary)::

        C[..., 0] = A[..., 0] * B[..., 0] + A[..., 1] * B[..., 1]
        C[..., 1] = A[..., 1] * B[..., 0] - A[..., 0] * B[..., 1]
    """

    op_domain = "com.microsoft"

    def _run(self, A, B):
        assert A.shape[-1] == 2, f"Expected last dim 2, got {A.shape}"
        assert B.shape[-1] == 2, f"Expected last dim 2, got {B.shape}"
        a_r, a_i = A[..., 0], A[..., 1]
        b_r, b_i = B[..., 0], B[..., 1]
        c_r = a_r * b_r + a_i * b_i
        c_i = a_i * b_r - a_r * b_i
        return (np.stack([c_r, c_i], axis=-1),)
