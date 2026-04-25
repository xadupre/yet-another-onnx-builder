import numpy as np
from onnx.reference.op_run import OpRun


class FusedMatMulActivation(OpRun):
    """Implements the ``com.microsoft.FusedMatMulActivation`` operator.

    Computes ``activation(alpha * A @ B)`` where ``activation`` is one of the
    supported element-wise activation functions.

    Supported values for ``activation``:

    - ``"Relu"`` — no extra parameters
    - ``"Tanh"`` — no extra parameters
    - ``"Sigmoid"`` — no extra parameters
    - ``"LeakyRelu"`` — uses ``activation_alpha`` (default 0.01)
    - ``"HardSigmoid"`` — uses ``activation_alpha`` (default 0.2) and
      ``activation_beta`` (default 0.5)
    """

    op_domain = "com.microsoft"

    def _run(
        self,
        A,
        B,
        activation: str = "Relu",
        activation_alpha: float = 0.0,
        activation_beta: float = 0.0,
        activation_gamma: float = 0.0,
        activation_axis: int = 0,
        alpha: float = 1.0,
        transA: int = 0,
        transB: int = 0,
        transBatchA: int = 0,
        transBatchB: int = 0,
    ):
        assert transBatchA == 0, f"Not implemented for transBatchA==1 and {A.shape}x{B.shape}"
        assert transBatchB == 0, f"Not implemented for transBatchB==1 and {A.shape}x{B.shape}"
        if transA:
            perm = list(range(len(A.shape)))
            dim = len(perm)
            perm[dim - 2], perm[dim - 1] = perm[dim - 1], perm[dim - 2]
            A = np.transpose(A, perm)
        if transB:
            perm = list(range(len(B.shape)))
            dim = len(perm)
            perm[dim - 2], perm[dim - 1] = perm[dim - 1], perm[dim - 2]
            B = np.transpose(B, perm)
        a = np.array(alpha, dtype=A.dtype)
        y = np.matmul(A, B) * a
        if activation == "Relu":
            y = np.maximum(y, 0)
        elif activation == "Tanh":
            y = np.tanh(y)
        elif activation == "Sigmoid":
            y = 1.0 / (1.0 + np.exp(-y))
        elif activation == "LeakyRelu":
            alpha_lr = activation_alpha if activation_alpha != 0.0 else 0.01
            y = np.where(y >= 0, y, alpha_lr * y)
        elif activation == "HardSigmoid":
            alpha_hs = activation_alpha if activation_alpha != 0.0 else 0.2
            beta_hs = activation_beta if activation_beta != 0.0 else 0.5
            y = np.clip(alpha_hs * y + beta_hs, 0.0, 1.0)
        else:
            raise NotImplementedError(f"Unsupported activation type: {activation!r}")
        return (y.astype(A.dtype),)
