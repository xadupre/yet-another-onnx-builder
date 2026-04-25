import numpy as np
from onnx.reference.op_run import OpRun


def _fast_gelu_core(x):
    """Computes the FastGelu activation.

    The formula is ``x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``.
    """
    cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
    return x * cdf


class FastGelu(OpRun):
    """Implements the ``com.microsoft.FastGelu`` operator.

    Applies the FastGelu activation, optionally adding a bias before the activation.
    """

    op_domain = "com.microsoft"

    def _run(self, X, bias=None):
        x = X if bias is None else (X + bias)
        return (_fast_gelu_core(x).astype(X.dtype),)
