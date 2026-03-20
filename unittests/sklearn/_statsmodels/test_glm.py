"""
Unit tests for the statsmodels GLM converter.
"""

import unittest
import numpy as np

from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_statsmodels
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


def _make_data(n=80, n_features=4, seed=42):
    """Return (X, y_gaussian, y_poisson, y_binomial) for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    y_gaussian = X @ np.array([0.5, -0.3, 0.8, 0.1]) + 2.0 + 0.1 * rng.standard_normal(n)
    y_poisson = rng.poisson(lam=np.exp(0.3 * X[:, 0] + 0.5), size=n)
    y_binomial = rng.binomial(1, 1 / (1 + np.exp(-0.5 * X[:, 0])), size=n).astype(float)
    return X, y_gaussian, y_poisson, y_binomial


@requires_sklearn("1.4")
@requires_statsmodels()
class TestStatsmodelsGLMConverter(ExtTestCase):
    """Tests for the StatsmodelsGLMWrapper ONNX converter."""

    _X, _y_gaussian, _y_poisson, _y_binomial = _make_data()

    def _check_glm(self, result, X, atol=1e-5):
        """Helper: convert to ONNX and compare predictions."""
        import statsmodels.api as sm
        from yobx.sklearn.statsmodels.glm import StatsmodelsGLMWrapper

        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)

            wrapper = StatsmodelsGLMWrapper(result)

            # Verify that the wrapper's predict() matches statsmodels
            exog_names = list(result.model.exog_names)
            if "const" in exog_names:
                X_for_sm = sm.add_constant(Xd)
            else:
                X_for_sm = Xd
            expected_sm = result.predict(X_for_sm).astype(dtype).reshape(-1, 1)
            wrapper_pred = wrapper.predict(Xd).astype(dtype).reshape(-1, 1)
            self.assertEqualArray(expected_sm, wrapper_pred, atol=atol)

            # Convert to ONNX
            onx = to_onnx(wrapper, (Xd,))
            op_types = {n.op_type for n in onx.graph.node}
            self.assertIn("Gemm", op_types)

            # Reference evaluator
            ref = ExtendedReferenceEvaluator(onx)
            result_ref = ref.run(None, {"X": Xd})[0]
            self.assertEqualArray(expected_sm, result_ref, atol=atol)

            # ONNX Runtime
            sess = self.check_ort(onx)
            ort_result = sess.run(None, {"X": Xd})[0]
            self.assertEqualArray(expected_sm, ort_result, atol=atol)

    def test_gaussian_identity_link(self):
        """Gaussian family with default Identity link."""
        import statsmodels.api as sm
        import statsmodels.genmod.families as families

        X_const = sm.add_constant(self._X)
        result = sm.GLM(self._y_gaussian, X_const, family=families.Gaussian()).fit()
        self._check_glm(result, self._X)

    def test_gaussian_no_intercept(self):
        """Gaussian family without intercept column."""
        import statsmodels.api as sm
        import statsmodels.genmod.families as families

        result = sm.GLM(self._y_gaussian, self._X, family=families.Gaussian()).fit()
        self._check_glm(result, self._X)

    def test_poisson_log_link(self):
        """Poisson family with default Log link."""
        import statsmodels.api as sm
        import statsmodels.genmod.families as families

        X_const = sm.add_constant(self._X)
        result = sm.GLM(self._y_poisson, X_const, family=families.Poisson()).fit()
        self._check_glm(result, self._X)

    def test_binomial_logit_link(self):
        """Binomial family with default Logit link."""
        import statsmodels.api as sm
        import statsmodels.genmod.families as families

        X_const = sm.add_constant(self._X)
        result = sm.GLM(self._y_binomial, X_const, family=families.Binomial()).fit()
        self._check_glm(result, self._X)

    def test_gaussian_log_link(self):
        """Gaussian family with explicit Log link."""
        import statsmodels.api as sm
        import statsmodels.genmod.families as families

        X_const = sm.add_constant(self._X)
        result = sm.GLM(
            self._y_poisson, X_const, family=families.Gaussian(families.links.Log())
        ).fit()
        self._check_glm(result, self._X)

    def test_gamma_inverse_power_link(self):
        """Gamma family with default InversePower link."""
        import statsmodels.api as sm
        import statsmodels.genmod.families as families

        # Gamma needs strictly positive target
        y = np.abs(self._y_gaussian) + 0.5
        X_const = sm.add_constant(self._X)
        result = sm.GLM(y, X_const, family=families.Gamma()).fit()
        self._check_glm(result, self._X)

    def test_gamma_log_link(self):
        """Gamma family with explicit Log link."""
        import statsmodels.api as sm
        import statsmodels.genmod.families as families

        y = np.abs(self._y_gaussian) + 0.5
        X_const = sm.add_constant(self._X)
        result = sm.GLM(y, X_const, family=families.Gamma(families.links.Log())).fit()
        self._check_glm(result, self._X)

    def test_wrapper_predict_matches_statsmodels(self):
        """Verify that StatsmodelsGLMWrapper.predict matches statsmodels prediction."""
        import statsmodels.api as sm
        import statsmodels.genmod.families as families
        from yobx.sklearn.statsmodels.glm import StatsmodelsGLMWrapper

        X_const = sm.add_constant(self._X)
        result = sm.GLM(self._y_poisson, X_const, family=families.Poisson()).fit()
        wrapper = StatsmodelsGLMWrapper(result)

        expected = result.predict(X_const)
        got = wrapper.predict(self._X)
        self.assertEqualArray(expected, got, atol=1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
