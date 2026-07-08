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


class Istft(OpRun):
    op_domain = "ai.onnx.complex"

    def _run(
        self,
        x,
        n_fft,
        hop_length,
        win_length,
        center,
        normalized,
        onesided,
        length,
        return_complex,
        window=None,
    ):
        import torch

        def _to_int(value):
            if isinstance(value, np.ndarray):
                return int(value.reshape((-1,))[0])
            return int(value)

        n_fft_i = _to_int(n_fft)
        hop_length_i = _to_int(hop_length)
        win_length_i = _to_int(win_length)
        center_b = bool(_to_int(center))
        normalized_b = bool(_to_int(normalized))
        onesided_i = _to_int(onesided)
        length_i = _to_int(length)
        return_complex_b = bool(_to_int(return_complex))

        tx = torch.from_numpy(x)
        twindow = torch.from_numpy(np.array(window, copy=True)) if window is not None else None
        output = torch.istft(
            tx,
            n_fft=n_fft_i,
            hop_length=hop_length_i,
            win_length=win_length_i,
            window=twindow,
            center=center_b,
            normalized=normalized_b,
            onesided=None if onesided_i < 0 else bool(onesided_i),
            length=None if length_i < 0 else length_i,
            return_complex=return_complex_b,
        )
        return (output.detach().cpu().numpy(),)


class FftC2r(OpRun):
    op_domain = "ai.onnx.complex"

    def _run(self, x, dim, normalization, last_dim_size):
        import torch

        def _to_int(value):
            if isinstance(value, np.ndarray):
                return int(value.reshape((-1,))[0])
            return int(value)

        def _to_dims(value):
            if isinstance(value, np.ndarray):
                return [int(v) for v in value.reshape((-1,)).tolist()]
            if isinstance(value, (list, tuple)):
                return [int(v) for v in value]
            return [int(value)]

        tx = torch.from_numpy(x)
        output = torch.ops.aten._fft_c2r.default(
            tx, _to_dims(dim), _to_int(normalization), _to_int(last_dim_size)
        )
        return (output.detach().cpu().numpy(),)


class FftIrfft2(OpRun):
    op_domain = "ai.onnx.complex"

    def _run(self, x, s, dim, norm):
        import torch

        def _to_int(value):
            if isinstance(value, np.ndarray):
                return int(value.reshape((-1,))[0])
            return int(value)

        def _to_dims(value):
            if isinstance(value, np.ndarray):
                return [int(v) for v in value.reshape((-1,)).tolist()]
            if isinstance(value, (list, tuple)):
                return [int(v) for v in value]
            return [int(value)]

        s_values = _to_dims(s)
        s_tuple = None if all(v < 0 for v in s_values) else tuple(s_values)
        dim_tuple = tuple(_to_dims(dim))
        norm_value = {0: "backward", 1: "forward", 2: "ortho"}[_to_int(norm)]

        tx = torch.from_numpy(x)
        output = torch.fft.irfft2(tx, s=s_tuple, dim=dim_tuple, norm=norm_value)
        return (output.detach().cpu().numpy(),)
