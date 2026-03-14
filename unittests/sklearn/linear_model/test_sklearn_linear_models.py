"""
Unit tests for the new sklearn linear model converters.

Covers:
- Plain linear regressors (LinearRegression, Ridge, Lasso, etc.)
- GLM regressors (TweedieRegressor, PoissonRegressor, GammaRegressor)
- Linear classifiers without predict_proba (RidgeClassifier, Perceptron, SGDClassifier)
"""

import unittest
import numpy as np
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    GammaRegressor,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskLasso,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    Perceptron,
    PoissonRegressor,
    QuantileRegressor,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
    TheilSenRegressor,
    TweedieRegressor,
)
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnLinearRegressorConverters(ExtTestCase):
    """Tests for plain linear regressor converters."""

    # --------------------------------------------------------------------- #
    # Single-output regressors                                               #
    # --------------------------------------------------------------------- #

    def _check_single_regressor(self, estimator, X, y, atol=1e-4):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            est = clone(estimator)
            est.fit(Xd, y)
            onx = to_onnx(est, (Xd,))

            op_types = [n.op_type for n in onx.graph.node]
            self.assertIn("Gemm", op_types)

            ref = ExtendedReferenceEvaluator(onx)
            result = ref.run(None, {"X": Xd})[0]
            expected = est.predict(Xd).astype(dtype).reshape(-1, 1)
            self.assertEqualArray(expected, result, atol=atol)

            sess = self.check_ort(onx)
            ort_result = sess.run(None, {"X": Xd})[0]
            self.assertEqualArray(expected, ort_result, atol=atol)

    def test_linear_regression(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(LinearRegression(), X, y)

    def test_ridge(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(Ridge(), X, y)

    def test_ridge_cv(self):
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.float32
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(RidgeCV(cv=3), X, y)

    def test_lasso(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(Lasso(), X, y)

    def test_lasso_cv(self):
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.float32
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(LassoCV(cv=3), X, y)

    def test_elastic_net(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(ElasticNet(), X, y)

    def test_elastic_net_cv(self):
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.float32
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(ElasticNetCV(cv=3), X, y)

    def test_lars(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(Lars(), X, y)

    def test_lars_cv(self):
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.float32
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(LarsCV(cv=3), X, y)

    def test_lasso_lars(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(LassoLars(), X, y)

    def test_lasso_lars_cv(self):
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.float32
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(LassoLarsCV(cv=3), X, y)

    def test_lasso_lars_ic(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5])
        self._check_single_regressor(LassoLarsIC(), X, y)

    def test_bayesian_ridge(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(BayesianRidge(), X, y)

    def test_ard_regression(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(ARDRegression(), X, y)

    def test_huber_regressor(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(HuberRegressor(), X, y)

    def test_theil_sen_regressor(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(TheilSenRegressor(), X, y)

    def test_quantile_regressor(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(QuantileRegressor(), X, y)

    def test_orthogonal_matching_pursuit(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(OrthogonalMatchingPursuit(), X, y)

    def test_orthogonal_matching_pursuit_cv(self):
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.float32
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        self._check_single_regressor(OrthogonalMatchingPursuitCV(cv=3), X, y)

    def test_sgd_regressor(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._check_single_regressor(SGDRegressor(random_state=0, max_iter=2000), X, y)

    # --------------------------------------------------------------------- #
    # Multi-output regressors                                                #
    # --------------------------------------------------------------------- #

    def test_multi_task_lasso(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [1.5, 2.5], [2.5, 3.5]])
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            m = MultiTaskLasso()
            m.fit(Xd, y)

            onx = to_onnx(m, (Xd,))
            op_types = [n.op_type for n in onx.graph.node]
            self.assertIn("Gemm", op_types)

            ref = ExtendedReferenceEvaluator(onx)
            result = ref.run(None, {"X": Xd})[0]
            expected = m.predict(Xd).astype(dtype)
            self.assertEqualArray(expected, result, atol=1e-4)

            sess = self.check_ort(onx)
            ort_result = sess.run(None, {"X": Xd})[0]
            self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_multi_task_elastic_net(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [1.5, 2.5], [2.5, 3.5]])
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            m = MultiTaskElasticNet()
            m.fit(Xd, y)

            onx = to_onnx(m, (Xd,))
            ref = ExtendedReferenceEvaluator(onx)
            result = ref.run(None, {"X": Xd})[0]
            expected = m.predict(Xd).astype(dtype)
            self.assertEqualArray(expected, result, atol=1e-4)

            sess = self.check_ort(onx)
            ort_result = sess.run(None, {"X": Xd})[0]
            self.assertEqualArray(expected, ort_result, atol=1e-4)


@requires_sklearn("1.4")
class TestSklearnGLMRegressorConverters(ExtTestCase):
    """Tests for GLM regressor converters (Tweedie, Poisson, Gamma)."""

    # Use strictly-positive targets for GLMs with log link
    _X = np.array(
        [[0.5, 1.0], [1.0, 2.0], [1.5, 3.0], [2.0, 4.0], [2.5, 5.0], [3.0, 6.0]], dtype=np.float32
    )
    _y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    def _check_glm(self, estimator, atol=1e-4):
        for dtype in (np.float32, np.float64):
            X = self._X.astype(dtype)
            y = self._y
            est = clone(estimator)
            est.fit(X, y)
            onx = to_onnx(est, (X,))

            op_types = [n.op_type for n in onx.graph.node]
            self.assertIn("Gemm", op_types)

            ref = ExtendedReferenceEvaluator(onx)
            result = ref.run(None, {"X": X})[0]
            expected = est.predict(X).astype(dtype).reshape(-1, 1)
            self.assertEqualArray(expected, result, atol=atol)

            sess = self.check_ort(onx)
            ort_result = sess.run(None, {"X": X})[0]
            self.assertEqualArray(expected, ort_result, atol=atol)

    def test_poisson_regressor(self):
        self._check_glm(PoissonRegressor())

    def test_gamma_regressor(self):
        self._check_glm(GammaRegressor())

    def test_tweedie_regressor_identity_link(self):
        # power=0 → identity link (Gaussian equivalent)
        self._check_glm(TweedieRegressor(power=0))

    def test_tweedie_regressor_log_link(self):
        # power=1 → log link (Poisson equivalent)
        self._check_glm(TweedieRegressor(power=1))

    def test_tweedie_regressor_log_link_explicit(self):
        # power=0 but with explicit 'log' link
        self._check_glm(TweedieRegressor(power=0, link="log"))

    def test_glm_op_type_identity(self):
        """TweedieRegressor with identity link should NOT produce Exp node."""
        for dtype in (np.float32, np.float64):
            X = self._X.astype(dtype)
            y = self._y
            m = TweedieRegressor(power=0)
            m.fit(X, y)
            onx = to_onnx(m, (X,))
            op_types = [n.op_type for n in onx.graph.node]
            self.assertIn("Gemm", op_types)
            self.assertNotIn("Exp", op_types)

    def test_glm_op_type_exp(self):
        """PoissonRegressor with log link should produce Exp node."""
        for dtype in (np.float32, np.float64):
            X = self._X.astype(dtype)
            y = self._y
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

    _X_multi = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
    _y_multi = np.array([0, 0, 1, 1, 2, 2])

    def _check_label_only_classifier(self, estimator, X, y):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            est = clone(estimator)
            est.fit(Xd, y)
            onx = to_onnx(est, (Xd,))

            # Should only produce a single (label) output
            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected_label = est.predict(Xd)
            self.assertEqualArray(expected_label, results[0])

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])

    def test_ridge_classifier_binary(self):
        self._check_label_only_classifier(RidgeClassifier(), self._X_bin, self._y_bin)

    def test_ridge_classifier_multiclass(self):
        self._check_label_only_classifier(RidgeClassifier(), self._X_multi, self._y_multi)

    def test_ridge_classifier_cv_binary(self):
        X = np.concatenate([self._X_bin] * 3, axis=0)
        y = np.concatenate([self._y_bin] * 3, axis=0)
        self._check_label_only_classifier(RidgeClassifierCV(cv=3), X, y)

    def test_ridge_classifier_cv_multiclass(self):
        X = np.concatenate([self._X_multi] * 3, axis=0)
        y = np.concatenate([self._y_multi] * 3, axis=0)
        self._check_label_only_classifier(RidgeClassifierCV(cv=3), X, y)

    def test_perceptron_binary(self):
        self._check_label_only_classifier(
            Perceptron(random_state=0, max_iter=1000), self._X_bin, self._y_bin
        )

    def test_perceptron_multiclass(self):
        self._check_label_only_classifier(
            Perceptron(random_state=0, max_iter=1000), self._X_multi, self._y_multi
        )

    def test_sgd_classifier_hinge_binary(self):
        """SGDClassifier with hinge loss (no predict_proba) → label only."""
        m = SGDClassifier(loss="hinge", random_state=0, max_iter=1000)
        self._check_label_only_classifier(m, self._X_bin, self._y_bin)

    def test_sgd_classifier_hinge_multiclass(self):
        """SGDClassifier with hinge loss (no predict_proba) → label only."""
        m = SGDClassifier(loss="hinge", random_state=0, max_iter=1000)
        self._check_label_only_classifier(m, self._X_multi, self._y_multi)

    def test_sgd_classifier_log_loss_binary(self):
        """SGDClassifier with log_loss (has predict_proba) → label + proba."""
        for dtype in (np.float32, np.float64):
            Xd = self._X_bin.astype(dtype)
            m = SGDClassifier(loss="log_loss", random_state=0, max_iter=1000)
            m.fit(Xd, self._y_bin)

            onx = to_onnx(m, (Xd,))

            # Should produce two outputs: label + probabilities
            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label, proba = results[0], results[1]

            expected_label = m.predict(Xd)
            expected_proba = m.predict_proba(Xd).astype(dtype)
            self.assertEqualArray(expected_label, label)
            self.assertEqualArray(expected_proba, proba, atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_sgd_classifier_log_loss_multiclass(self):
        """SGDClassifier with log_loss multiclass → label + proba."""
        for dtype in (np.float32, np.float64):
            Xd = self._X_multi.astype(dtype)
            m = SGDClassifier(loss="log_loss", random_state=0, max_iter=1000)
            m.fit(Xd, self._y_multi)

            onx = to_onnx(m, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label, proba = results[0], results[1]

            expected_label = m.predict(Xd)
            expected_proba = m.predict_proba(Xd).astype(dtype)
            self.assertEqualArray(expected_label, label)
            self.assertEqualArray(expected_proba, proba, atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_ridge_classifier_in_pipeline(self):
        """RidgeClassifier at the end of a Pipeline."""
        for dtype in (np.float32, np.float64):
            Xd = self._X_bin.astype(dtype)
            y = self._y_bin
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier())])
            pipe.fit(Xd, y)

            onx = to_onnx(pipe, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected_label = pipe.predict(Xd)
            self.assertEqualArray(expected_label, results[0])

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
