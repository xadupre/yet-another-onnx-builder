"""
Unit tests for the new sklearn linear model converters.

Covers:
- Plain linear regressors (LinearRegression, Ridge, Lasso, etc.)
- GLM regressors (TweedieRegressor, PoissonRegressor, GammaRegressor)
- Linear classifiers without predict_proba (RidgeClassifier, Perceptron, SGDClassifier)
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestSklearnLinearRegressorConverters(ExtTestCase):
    """Tests for plain linear regressor converters."""

    # --------------------------------------------------------------------- #
    # Single-output regressors                                               #
    # --------------------------------------------------------------------- #

    def _check_single_regressor(self, estimator, X, y, atol=1e-4):
        from yobx.sklearn import to_onnx

        estimator.fit(X, y)
        onx = to_onnx(estimator, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gemm", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = estimator.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, result, atol=atol)

    def test_linear_regression(self):
        from sklearn.linear_model import LinearRegression

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(LinearRegression(), X, y)

    def test_ridge(self):
        from sklearn.linear_model import Ridge

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(Ridge(), X, y)

    def test_ridge_cv(self):
        from sklearn.linear_model import RidgeCV

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]],
            dtype=np.float32,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(RidgeCV(cv=3), X, y)

    def test_lasso(self):
        from sklearn.linear_model import Lasso

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(Lasso(), X, y)

    def test_lasso_cv(self):
        from sklearn.linear_model import LassoCV

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]],
            dtype=np.float32,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(LassoCV(cv=3), X, y)

    def test_elastic_net(self):
        from sklearn.linear_model import ElasticNet

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(ElasticNet(), X, y)

    def test_elastic_net_cv(self):
        from sklearn.linear_model import ElasticNetCV

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]],
            dtype=np.float32,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(ElasticNetCV(cv=3), X, y)

    def test_lars(self):
        from sklearn.linear_model import Lars

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(Lars(), X, y)

    def test_lars_cv(self):
        from sklearn.linear_model import LarsCV

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]],
            dtype=np.float32,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(LarsCV(cv=3), X, y)

    def test_lasso_lars(self):
        from sklearn.linear_model import LassoLars

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(LassoLars(), X, y)

    def test_lasso_lars_cv(self):
        from sklearn.linear_model import LassoLarsCV

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]],
            dtype=np.float32,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(LassoLarsCV(cv=3), X, y)

    def test_lasso_lars_ic(self):
        from sklearn.linear_model import LassoLarsIC

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]],
            dtype=np.float32,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5])
        self._check_single_regressor(LassoLarsIC(), X, y)

    def test_bayesian_ridge(self):
        from sklearn.linear_model import BayesianRidge

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(BayesianRidge(), X, y)

    def test_ard_regression(self):
        from sklearn.linear_model import ARDRegression

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(ARDRegression(), X, y)

    def test_huber_regressor(self):
        from sklearn.linear_model import HuberRegressor

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(HuberRegressor(), X, y)

    def test_theil_sen_regressor(self):
        from sklearn.linear_model import TheilSenRegressor

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(TheilSenRegressor(), X, y)

    def test_quantile_regressor(self):
        from sklearn.linear_model import QuantileRegressor

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(QuantileRegressor(), X, y)

    def test_orthogonal_matching_pursuit(self):
        from sklearn.linear_model import OrthogonalMatchingPursuit

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(OrthogonalMatchingPursuit(), X, y)

    def test_orthogonal_matching_pursuit_cv(self):
        from sklearn.linear_model import OrthogonalMatchingPursuitCV

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]],
            dtype=np.float32,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(OrthogonalMatchingPursuitCV(cv=3), X, y)

    def test_sgd_regressor(self):
        from sklearn.linear_model import SGDRegressor

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(SGDRegressor(random_state=0, max_iter=2000), X, y)

    # --------------------------------------------------------------------- #
    # Multi-output regressors                                                #
    # --------------------------------------------------------------------- #

    def test_multi_task_lasso(self):
        from sklearn.linear_model import MultiTaskLasso
        from yobx.sklearn import to_onnx

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]],
            dtype=np.float32,
        )
        y = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [1.5, 2.5], [2.5, 3.5]])
        m = MultiTaskLasso()
        m.fit(X, y)

        onx = to_onnx(m, (X,))
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gemm", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = m.predict(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

    def test_multi_task_elastic_net(self):
        from sklearn.linear_model import MultiTaskElasticNet
        from yobx.sklearn import to_onnx

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]],
            dtype=np.float32,
        )
        y = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [1.5, 2.5], [2.5, 3.5]])
        m = MultiTaskElasticNet()
        m.fit(X, y)

        onx = to_onnx(m, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = m.predict(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)


@requires_sklearn("1.4")
class TestSklearnGLMRegressorConverters(ExtTestCase):
    """Tests for GLM regressor converters (Tweedie, Poisson, Gamma)."""

    # Use strictly-positive targets for GLMs with log link
    _X = np.array(
        [[0.5, 1.0], [1.0, 2.0], [1.5, 3.0], [2.0, 4.0], [2.5, 5.0], [3.0, 6.0]],
        dtype=np.float32,
    )
    _y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    def _check_glm(self, estimator, atol=1e-4):
        from yobx.sklearn import to_onnx

        X, y = self._X, self._y
        estimator.fit(X, y)
        onx = to_onnx(estimator, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gemm", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = estimator.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, result, atol=atol)

    def test_poisson_regressor(self):
        from sklearn.linear_model import PoissonRegressor

        self._check_glm(PoissonRegressor())

    def test_gamma_regressor(self):
        from sklearn.linear_model import GammaRegressor

        self._check_glm(GammaRegressor())

    def test_tweedie_regressor_identity_link(self):
        from sklearn.linear_model import TweedieRegressor

        # power=0 → identity link (Gaussian equivalent)
        self._check_glm(TweedieRegressor(power=0))

    def test_tweedie_regressor_log_link(self):
        from sklearn.linear_model import TweedieRegressor

        # power=1 → log link (Poisson equivalent)
        self._check_glm(TweedieRegressor(power=1))

    def test_tweedie_regressor_log_link_explicit(self):
        from sklearn.linear_model import TweedieRegressor

        # power=0 but with explicit 'log' link
        self._check_glm(TweedieRegressor(power=0, link="log"))

    def test_glm_op_type_identity(self):
        """TweedieRegressor with identity link should NOT produce Exp node."""
        from sklearn.linear_model import TweedieRegressor
        from yobx.sklearn import to_onnx

        X, y = self._X, self._y
        m = TweedieRegressor(power=0)
        m.fit(X, y)
        onx = to_onnx(m, (X,))
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gemm", op_types)
        self.assertNotIn("Exp", op_types)

    def test_glm_op_type_exp(self):
        """PoissonRegressor with log link should produce Exp node."""
        from sklearn.linear_model import PoissonRegressor
        from yobx.sklearn import to_onnx

        X, y = self._X, self._y
        m = PoissonRegressor()
        m.fit(X, y)
        onx = to_onnx(m, (X,))
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Exp", op_types)


@requires_sklearn("1.4")
class TestSklearnLinearClassifierConverters(ExtTestCase):
    """Tests for linear classifiers without predict_proba."""

    _X_bin = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    _y_bin = np.array([0, 0, 1, 1])

    _X_multi = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32
    )
    _y_multi = np.array([0, 0, 1, 1, 2, 2])

    def _check_label_only_classifier(self, estimator, X, y):
        from yobx.sklearn import to_onnx

        estimator.fit(X, y)
        onx = to_onnx(estimator, (X,))

        # Should only produce a single (label) output
        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        self.assertEqualArray(estimator.predict(X), results[0])

    def test_ridge_classifier_binary(self):
        from sklearn.linear_model import RidgeClassifier

        self._check_label_only_classifier(RidgeClassifier(), self._X_bin, self._y_bin)

    def test_ridge_classifier_multiclass(self):
        from sklearn.linear_model import RidgeClassifier

        self._check_label_only_classifier(
            RidgeClassifier(), self._X_multi, self._y_multi
        )

    def test_ridge_classifier_cv_binary(self):
        from sklearn.linear_model import RidgeClassifierCV

        X = np.concatenate([self._X_bin] * 3, axis=0)
        y = np.concatenate([self._y_bin] * 3, axis=0)
        self._check_label_only_classifier(RidgeClassifierCV(cv=3), X, y)

    def test_ridge_classifier_cv_multiclass(self):
        from sklearn.linear_model import RidgeClassifierCV

        X = np.concatenate([self._X_multi] * 3, axis=0)
        y = np.concatenate([self._y_multi] * 3, axis=0)
        self._check_label_only_classifier(RidgeClassifierCV(cv=3), X, y)

    def test_perceptron_binary(self):
        from sklearn.linear_model import Perceptron

        self._check_label_only_classifier(
            Perceptron(random_state=0, max_iter=1000), self._X_bin, self._y_bin
        )

    def test_perceptron_multiclass(self):
        from sklearn.linear_model import Perceptron

        self._check_label_only_classifier(
            Perceptron(random_state=0, max_iter=1000), self._X_multi, self._y_multi
        )

    def test_sgd_classifier_hinge_binary(self):
        """SGDClassifier with hinge loss (no predict_proba) → label only."""
        from sklearn.linear_model import SGDClassifier

        m = SGDClassifier(loss="hinge", random_state=0, max_iter=1000)
        self._check_label_only_classifier(m, self._X_bin, self._y_bin)

    def test_sgd_classifier_hinge_multiclass(self):
        """SGDClassifier with hinge loss (no predict_proba) → label only."""
        from sklearn.linear_model import SGDClassifier

        m = SGDClassifier(loss="hinge", random_state=0, max_iter=1000)
        self._check_label_only_classifier(m, self._X_multi, self._y_multi)

    def test_sgd_classifier_log_loss_binary(self):
        """SGDClassifier with log_loss (has predict_proba) → label + proba."""
        from sklearn.linear_model import SGDClassifier
        from yobx.sklearn import to_onnx

        m = SGDClassifier(loss="log_loss", random_state=0, max_iter=1000)
        m.fit(self._X_bin, self._y_bin)

        onx = to_onnx(m, (self._X_bin,))

        # Should produce two outputs: label + probabilities
        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_bin})
        label, proba = results[0], results[1]

        self.assertEqualArray(m.predict(self._X_bin), label)
        self.assertEqualArray(
            m.predict_proba(self._X_bin).astype(np.float32), proba, atol=1e-5
        )

    def test_sgd_classifier_log_loss_multiclass(self):
        """SGDClassifier with log_loss multiclass → label + proba."""
        from sklearn.linear_model import SGDClassifier
        from yobx.sklearn import to_onnx

        m = SGDClassifier(loss="log_loss", random_state=0, max_iter=1000)
        m.fit(self._X_multi, self._y_multi)

        onx = to_onnx(m, (self._X_multi,))

        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_multi})
        label, proba = results[0], results[1]

        self.assertEqualArray(m.predict(self._X_multi), label)
        self.assertEqualArray(
            m.predict_proba(self._X_multi).astype(np.float32), proba, atol=1e-5
        )

    def test_ridge_classifier_in_pipeline(self):
        """RidgeClassifier at the end of a Pipeline."""
        from sklearn.linear_model import RidgeClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        X, y = self._X_bin, self._y_bin
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        self.assertEqualArray(pipe.predict(X), results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
