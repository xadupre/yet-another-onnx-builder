"""
Converter for :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`.
"""

from typing import Dict, List

import numpy as np
from sklearn.base import BaseEstimator

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


class StatsmodelsGLMWrapper(BaseEstimator):
    """
    Wraps a fitted :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    as a :epkg:`scikit-learn`-compatible estimator so it can be converted to ONNX via
    :func:`yobx.sklearn.to_onnx`.

    The class extracts the fitted parameters, intercept, and link function from the
    statsmodels result at construction time.  The :meth:`predict` method replicates
    statsmodels' prediction logic without requiring the full design matrix (i.e. users
    pass the raw feature matrix *X*, **without** any constant/intercept column).

    Supported link functions:

    * **Identity** (Gaussian default) – pass-through
    * **Log** (Poisson / NegativeBinomial / Tweedie default) – ``exp(eta)``
    * **Logit** (Binomial default) – ``sigmoid(eta)``
    * **Power** (general) – ``eta ** (1 / power)``

      * **InversePower** (*power = -1*, Gamma default) – ``1 / eta``
      * **Sqrt** (*power = 0.5*) – ``eta ** 2``
      * **InverseSquared** (*power = -2*, InverseGaussian default) – ``eta ** (-0.5)``

    Example usage::

        import statsmodels.api as sm
        import statsmodels.genmod.families as families
        import numpy as np
        from yobx.sklearn import to_onnx
        from yobx.sklearn.statsmodels import StatsmodelsGLMWrapper

        X_train = np.column_stack([np.ones(100), np.random.randn(100, 3)])
        y_train = np.random.poisson(lam=2.0, size=100)
        result = sm.GLM(y_train, X_train, family=families.Poisson()).fit()

        wrapper = StatsmodelsGLMWrapper(result)
        # X_raw is the feature matrix WITHOUT the constant column
        X_raw = X_train[:, 1:]
        onx = to_onnx(wrapper, (X_raw.astype(np.float32),))

    :param glm_result: a fitted
        :class:`~statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    """

    def __init__(self, glm_result):
        self.glm_result = glm_result
        self._extract_params()

    def _extract_params(self):
        """Extract model coefficients, intercept, and link from the fitted result."""
        result = self.glm_result
        exog_names = list(result.model.exog_names)

        # Detect whether a constant (intercept) column was included in the design matrix.
        self.has_intercept_ = "const" in exog_names

        params = result.params.copy()

        if self.has_intercept_:
            const_idx = exog_names.index("const")
            self.intercept_ = np.atleast_1d(params[const_idx])
            mask = np.ones(len(params), dtype=bool)
            mask[const_idx] = False
            self.coef_ = params[mask].reshape(1, -1)
        else:
            self.intercept_ = np.zeros(1, dtype=params.dtype)
            self.coef_ = params.reshape(1, -1)

        self.link_ = result.family.link
        self.link_cls_name_ = type(result.family.link).__name__

    def fit(self, X=None, y=None, **fit_params):
        """
        No-op placeholder required by the :epkg:`scikit-learn` estimator API.

        :class:`StatsmodelsGLMWrapper` wraps an already-fitted statsmodels result,
        so this method is intentionally empty and returns ``self`` unchanged.
        """
        return self

    def predict(self, X):
        """
        Predict using the GLM model.

        Computes ``link.inverse(X @ coef.T + intercept)`` matching statsmodels'
        :meth:`~statsmodels.genmod.generalized_linear_model.GLMResultsWrapper.predict`.

        :param X: raw feature matrix of shape ``(n_samples, n_features)`` **without**
            any constant column
        :return: predicted values of shape ``(n_samples,)``
        """
        linpred = (X @ self.coef_.T + self.intercept_).ravel()
        return self.link_.inverse(linpred)


def _inverse_link_onnx(g, link, eta: str, itype: int, name: str, outputs: List[str]) -> str:
    """
    Apply the ONNX equivalent of ``link.inverse(eta)``.

    :param g: graph builder
    :param link: statsmodels link object
    :param eta: name of the linear-predictor tensor
    :param itype: ONNX element type of *eta*
    :param name: node-name prefix
    :param outputs: desired output tensor names
    :return: output tensor name
    :raises NotImplementedError: for unsupported link functions
    """
    from statsmodels.genmod.families import links as sm_links

    link_cls = type(link).__name__

    if isinstance(link, sm_links.Identity):
        return g.op.Identity(eta, name=name, outputs=outputs)

    if isinstance(link, sm_links.Log):
        return g.op.Exp(eta, name=name, outputs=outputs)

    if isinstance(link, sm_links.Logit):
        return g.op.Sigmoid(eta, name=name, outputs=outputs)

    if isinstance(link, sm_links.Power):
        # Power link: g(mu) = mu^p  →  inverse: mu = eta^(1/p)
        p = float(link.power)
        if p == 0:
            raise NotImplementedError(
                "Power link with exponent 0 is not invertible and cannot be converted to ONNX."
            )
        inv_p = 1.0 / p
        exp_const = g.op.Constant(
            value_float=float(inv_p), name=f"{name}_exp", outputs=[f"{name}_exp_cst"]
        )
        return g.op.Pow(eta, exp_const, name=name, outputs=outputs)

    raise NotImplementedError(
        f"Unsupported link function {link_cls!r} for statsmodels GLM ONNX conversion. "
        "Supported links: Identity, Log, Logit, Power (including InversePower, Sqrt, "
        "InverseSquared)."
    )


@register_sklearn_converter(StatsmodelsGLMWrapper)
def statsmodels_glm_converter(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: StatsmodelsGLMWrapper,
    X: str,
    name: str = "statsmodels_glm",
) -> str:
    """
    Converts a :class:`StatsmodelsGLMWrapper` into ONNX.

    The converter implements the GLM prediction formula::

        eta = X @ coef.T + intercept
        mu  = link⁻¹(eta)

    where *link* is the link function stored on the fitted statsmodels model.

    Graph structure:

    .. code-block:: text

        X  ──Gemm(coef, intercept, transB=1)──► eta
                                                  │
                                      link⁻¹(·) ──►  mu  (output)

    Supported link functions:

    * **Identity** – ``Identity`` node (pass-through)
    * **Log** – ``Exp``
    * **Logit** – ``Sigmoid``
    * **Power(p)** – ``Pow(eta, 1/p)``

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a :class:`StatsmodelsGLMWrapper` wrapping a fitted GLM result
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    :raises NotImplementedError: when an unsupported link function is encountered
    """
    assert isinstance(
        estimator, StatsmodelsGLMWrapper
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    coef = estimator.coef_.astype(dtype)
    intercept = estimator.intercept_.astype(dtype)

    # Ensure coef is 2D: (1, n_features) for Gemm with transB=1.
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    # Linear predictor: X @ coef.T + intercept  →  (N, 1)
    eta = g.op.Gemm(X, coef, intercept, transB=1, name=f"{name}_gemm")
    assert isinstance(eta, str)

    # Inverse link function
    result = _inverse_link_onnx(g, estimator.link_, eta, itype, name, outputs)

    assert isinstance(result, str)
    if not sts:
        g.set_type(result, itype)
    return result
