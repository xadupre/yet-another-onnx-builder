"""
Tests that check :func:`yobx.sklearn.to_onnx` is compatible with the test
patterns used in the *sklearn-onnx* (``skl2onnx``) project.

Each test follows the same structure as the sklearn-onnx test suite:

1. Build a small dataset.
2. Fit a scikit-learn estimator.
3. Convert with :func:`yobx.sklearn.to_onnx`.
4. Validate the produced :class:`onnx.ModelProto` with ``onnx.checker``.
5. Run the model with ``onnxruntime`` and assert numerical equivalence with
   scikit-learn's own predictions.

A subset of tests also compares yobx's output with the output produced by
``skl2onnx.convert_sklearn``, using the same input data, to confirm that both
converters are numerically equivalent for the models they share.

The companion CI workflow ``ci_sklearn_onnx.yml`` clones the sklearn-onnx
repository and runs its test suite end-to-end against yobx (via a monkey-patch
conftest), so the tests here focus on the models covered by yobx and on the
comparison with skl2onnx.
"""

import unittest
import numpy as np
import onnx
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.sklearn import to_onnx


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binary(n=40, n_features=4, seed=0):
    rng = _rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


def _make_multiclass(n=60, n_features=4, n_classes=3, seed=1):
    rng = _rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (np.abs(X[:, 0] * n_classes) % n_classes).astype(np.int64)
    return X, y


def _make_regression(n=40, n_features=4, seed=2):
    rng = _rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = X @ rng.standard_normal(n_features).astype(np.float32)
    return X, y


def _make_multitask_regression(n=40, n_features=4, n_targets=3, seed=3):
    rng = _rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = X @ rng.standard_normal((n_features, n_targets)).astype(np.float32)
    return X, y


@requires_sklearn("1.4")
class TestSklearnOnnxLinearModels(ExtTestCase):
    """sklearn-onnx-style tests for linear model converters."""

    # ------------------------------------------------------------------
    # Regressors
    # ------------------------------------------------------------------

    def _check_regressor(self, estimator, X, y, atol=1e-4):
        estimator.fit(X, y)
        onx = to_onnx(estimator, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (pred,) = sess.run(None, {"X": X})
        expected = estimator.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, pred, atol=atol)

    def test_linear_regression(self):
        from sklearn.linear_model import LinearRegression

        X, y = _make_regression()
        self._check_regressor(LinearRegression(), X, y)

    def test_ridge(self):
        from sklearn.linear_model import Ridge

        X, y = _make_regression()
        self._check_regressor(Ridge(alpha=1.0), X, y)

    def test_lasso(self):
        from sklearn.linear_model import Lasso

        X, y = _make_regression()
        self._check_regressor(Lasso(alpha=0.01), X, y)

    def test_elastic_net(self):
        from sklearn.linear_model import ElasticNet

        X, y = _make_regression()
        self._check_regressor(ElasticNet(alpha=0.01), X, y)

    def test_bayesian_ridge(self):
        from sklearn.linear_model import BayesianRidge

        X, y = _make_regression()
        self._check_regressor(BayesianRidge(), X, y)

    def test_huber_regressor(self):
        from sklearn.linear_model import HuberRegressor

        X, y = _make_regression()
        self._check_regressor(HuberRegressor(), X, y)

    def test_sgd_regressor(self):
        from sklearn.linear_model import SGDRegressor

        X, y = _make_regression()
        self._check_regressor(SGDRegressor(max_iter=500, random_state=0), X, y, atol=1e-3)

    def test_quantile_regressor(self):
        from sklearn.linear_model import QuantileRegressor

        X, y = _make_regression()
        self._check_regressor(QuantileRegressor(alpha=0.5, solver="highs"), X, y, atol=1e-4)

    def test_multitask_lasso(self):
        from sklearn.linear_model import MultiTaskLasso

        X, y = _make_multitask_regression()
        est = MultiTaskLasso(alpha=0.01)
        est.fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (pred,) = sess.run(None, {"X": X})
        expected = est.predict(X).astype(np.float32)
        self.assertEqualArray(expected, pred, atol=1e-4)

    def test_multitask_elasticnet(self):
        from sklearn.linear_model import MultiTaskElasticNet

        X, y = _make_multitask_regression()
        est = MultiTaskElasticNet(alpha=0.01)
        est.fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (pred,) = sess.run(None, {"X": X})
        expected = est.predict(X).astype(np.float32)
        self.assertEqualArray(expected, pred, atol=1e-4)

    # ------------------------------------------------------------------
    # Classifiers
    # ------------------------------------------------------------------

    def _check_classifier(self, estimator, X, y, atol=1e-4):
        estimator.fit(X, y)
        onx = to_onnx(estimator, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        label, proba = sess.run(None, {"X": X})
        expected_label = estimator.predict(X).astype(np.int64)
        expected_proba = estimator.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=atol)

    def test_logistic_regression_binary(self):
        from sklearn.linear_model import LogisticRegression

        X, y = _make_binary()
        self._check_classifier(LogisticRegression(random_state=0), X, y)

    def test_logistic_regression_multiclass(self):
        from sklearn.linear_model import LogisticRegression

        X, y = _make_multiclass()
        self._check_classifier(LogisticRegression(random_state=0, max_iter=300), X, y)

    def test_ridge_classifier(self):
        from sklearn.linear_model import RidgeClassifier

        X, y = _make_binary()
        est = RidgeClassifier()
        est.fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (label,) = sess.run(None, {"X": X})
        expected = est.predict(X).astype(np.int64)
        self.assertEqualArray(expected, label)

    def test_sgd_classifier(self):
        from sklearn.linear_model import SGDClassifier

        X, y = _make_binary()
        est = SGDClassifier(loss="log_loss", max_iter=500, random_state=0)
        self._check_classifier(est, X, y)

    def test_perceptron(self):
        from sklearn.linear_model import Perceptron

        X, y = _make_binary()
        est = Perceptron(max_iter=100, random_state=0)
        est.fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (label,) = sess.run(None, {"X": X})
        expected = est.predict(X).astype(np.int64)
        self.assertEqualArray(expected, label)

    def test_passive_aggressive_classifier(self):
        from sklearn.linear_model import PassiveAggressiveClassifier

        X, y = _make_binary()
        est = PassiveAggressiveClassifier(max_iter=100, random_state=0)
        est.fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (label,) = sess.run(None, {"X": X})
        expected = est.predict(X).astype(np.int64)
        self.assertEqualArray(expected, label)


@requires_sklearn("1.4")
class TestSklearnOnnxTreeModels(ExtTestCase):
    """sklearn-onnx-style tests for decision-tree-based converters."""

    def _check_classifier(self, estimator, X, y, atol=1e-4):
        estimator.fit(X, y)
        onx = to_onnx(estimator, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        label, proba = sess.run(None, {"X": X})
        self.assertEqualArray(estimator.predict(X).astype(np.int64), label)
        self.assertEqualArray(estimator.predict_proba(X).astype(np.float32), proba, atol=atol)

    def _check_regressor(self, estimator, X, y, atol=1e-4):
        estimator.fit(X, y)
        onx = to_onnx(estimator, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (pred,) = sess.run(None, {"X": X})
        expected = estimator.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, pred, atol=atol)

    def test_decision_tree_classifier_binary(self):
        from sklearn.tree import DecisionTreeClassifier

        X, y = _make_binary()
        self._check_classifier(DecisionTreeClassifier(random_state=0), X, y)

    def test_decision_tree_classifier_multiclass(self):
        from sklearn.tree import DecisionTreeClassifier

        X, y = _make_multiclass()
        self._check_classifier(DecisionTreeClassifier(random_state=0), X, y)

    def test_decision_tree_regressor(self):
        from sklearn.tree import DecisionTreeRegressor

        X, y = _make_regression()
        self._check_regressor(DecisionTreeRegressor(random_state=0), X, y)

    def test_random_forest_classifier_binary(self):
        from sklearn.ensemble import RandomForestClassifier

        X, y = _make_binary()
        self._check_classifier(RandomForestClassifier(n_estimators=5, random_state=0), X, y)

    def test_random_forest_classifier_multiclass(self):
        from sklearn.ensemble import RandomForestClassifier

        X, y = _make_multiclass()
        self._check_classifier(RandomForestClassifier(n_estimators=5, random_state=0), X, y)

    def test_random_forest_regressor(self):
        from sklearn.ensemble import RandomForestRegressor

        X, y = _make_regression()
        self._check_regressor(RandomForestRegressor(n_estimators=5, random_state=0), X, y)

    def test_hist_gradient_boosting_classifier_binary(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        X, y = _make_binary(n=80)
        self._check_classifier(HistGradientBoostingClassifier(max_iter=20, random_state=0), X, y)

    def test_hist_gradient_boosting_classifier_multiclass(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        X, y = _make_multiclass(n=90)
        self._check_classifier(HistGradientBoostingClassifier(max_iter=20, random_state=0), X, y)

    def test_hist_gradient_boosting_regressor(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        X, y = _make_regression(n=80)
        self._check_regressor(HistGradientBoostingRegressor(max_iter=20, random_state=0), X, y)


@requires_sklearn("1.4")
class TestSklearnOnnxPreprocessing(ExtTestCase):
    """sklearn-onnx-style tests for preprocessing converters."""

    def test_standard_scaler_float32(self):
        from sklearn.preprocessing import StandardScaler

        X, _ = _make_regression()
        est = StandardScaler().fit(X)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (out,) = sess.run(None, {"X": X})
        expected = est.transform(X).astype(np.float32)
        self.assertEqualArray(expected, out, atol=1e-5)

    def test_standard_scaler_float64(self):
        from sklearn.preprocessing import StandardScaler

        X, _ = _make_regression()
        X = X.astype(np.float64)
        est = StandardScaler().fit(X)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (out,) = sess.run(None, {"X": X})
        expected = est.transform(X).astype(np.float64)
        self.assertEqualArray(expected, out, atol=1e-10)

    def test_min_max_scaler(self):
        from sklearn.preprocessing import MinMaxScaler

        X, _ = _make_regression()
        est = MinMaxScaler().fit(X)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (out,) = sess.run(None, {"X": X})
        expected = est.transform(X).astype(np.float32)
        self.assertEqualArray(expected, out, atol=1e-5)

    def test_pca(self):
        from sklearn.decomposition import PCA

        X, _ = _make_regression(n=60)
        est = PCA(n_components=2).fit(X)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (out,) = sess.run(None, {"X": X})
        expected = est.transform(X).astype(np.float32)
        self.assertEqualArray(expected, out, atol=1e-5)


@requires_sklearn("1.4")
class TestSklearnOnnxNeighbors(ExtTestCase):
    """sklearn-onnx-style tests for neighbor-based converters."""

    def test_knn_classifier(self):
        from sklearn.neighbors import KNeighborsClassifier

        X, y = _make_binary(n=50)
        est = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        label, _ = sess.run(None, {"X": X})
        self.assertEqualArray(est.predict(X).astype(np.int64), label)

    def test_knn_regressor(self):
        from sklearn.neighbors import KNeighborsRegressor

        X, y = _make_regression(n=50)
        est = KNeighborsRegressor(n_neighbors=3).fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (pred,) = sess.run(None, {"X": X})
        expected = est.predict(X).astype(np.float32)
        self.assertEqualArray(expected, pred, atol=1e-5)


@requires_sklearn("1.4")
class TestSklearnOnnxClusterAndDecomposition(ExtTestCase):
    """sklearn-onnx-style tests for cluster and decomposition converters."""

    def test_kmeans(self):
        from sklearn.cluster import KMeans

        X, _ = _make_regression(n=60)
        est = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        label, _ = sess.run(None, {"X": X})
        self.assertEqualArray(est.predict(X).astype(np.int64), label)

    def test_lda(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        X, y = _make_binary(n=60)
        est = LinearDiscriminantAnalysis().fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        label, proba = sess.run(None, {"X": X})
        self.assertEqualArray(est.predict(X).astype(np.int64), label)
        self.assertEqualArray(est.predict_proba(X).astype(np.float32), proba, atol=1e-5)


@requires_sklearn("1.4")
class TestSklearnOnnxNeuralNetworks(ExtTestCase):
    """sklearn-onnx-style tests for neural network converters."""

    def test_mlp_classifier_binary(self):
        from sklearn.neural_network import MLPClassifier

        X, y = _make_binary()
        est = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=0)
        est.fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        label, proba = sess.run(None, {"X": X})
        self.assertEqualArray(est.predict(X).astype(np.int64), label)
        self.assertEqualArray(est.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_mlp_classifier_multiclass(self):
        from sklearn.neural_network import MLPClassifier

        X, y = _make_multiclass()
        est = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=0)
        est.fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        label, proba = sess.run(None, {"X": X})
        self.assertEqualArray(est.predict(X).astype(np.int64), label)
        self.assertEqualArray(est.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_mlp_regressor(self):
        from sklearn.neural_network import MLPRegressor

        X, y = _make_regression()
        est = MLPRegressor(hidden_layer_sizes=(8,), max_iter=500, random_state=0)
        est.fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (pred,) = sess.run(None, {"X": X})
        expected = est.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, pred, atol=1e-4)


@requires_sklearn("1.4")
class TestSklearnOnnxPipeline(ExtTestCase):
    """sklearn-onnx-style tests for Pipeline and ColumnTransformer converters."""

    def test_pipeline_scaler_lr(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        X, y = _make_binary()
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=0))]
        )
        pipe.fit(X, y)
        onx = to_onnx(pipe, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        label, proba = sess.run(None, {"X": X})
        self.assertEqualArray(pipe.predict(X).astype(np.int64), label)
        self.assertEqualArray(pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_pipeline_scaler_rf(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        X, y = _make_multiclass()
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
            ]
        )
        pipe.fit(X, y)
        onx = to_onnx(pipe, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        label, proba = sess.run(None, {"X": X})
        self.assertEqualArray(pipe.predict(X).astype(np.int64), label)
        self.assertEqualArray(pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_pipeline_pca_regressor(self):
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression

        X, y = _make_regression(n=60)
        pipe = Pipeline([("pca", PCA(n_components=2)), ("reg", LinearRegression())])
        pipe.fit(X, y)
        onx = to_onnx(pipe, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (pred,) = sess.run(None, {"X": X})
        expected = pipe.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, pred, atol=1e-4)

    def test_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        rng = _rng(4)
        X = rng.standard_normal((40, 6)).astype(np.float32)
        ct = ColumnTransformer(
            [("std", StandardScaler(), [0, 1, 2]), ("mm", MinMaxScaler(), [3, 4, 5])]
        )
        ct.fit(X)
        onx = to_onnx(ct, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        (out,) = sess.run(None, {"X": X})
        expected = ct.transform(X).astype(np.float32)
        self.assertEqualArray(expected, out, atol=1e-5)


@requires_sklearn("1.4")
class TestSklearnOnnxMulticlass(ExtTestCase):
    """sklearn-onnx-style tests for multiclass wrappers."""

    def test_one_vs_rest_classifier(self):
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.linear_model import LogisticRegression

        X, y = _make_multiclass()
        est = OneVsRestClassifier(LogisticRegression(random_state=0, max_iter=300)).fit(X, y)
        onx = to_onnx(est, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(onx))
        sess = self.check_ort(onx)
        label, proba = sess.run(None, {"X": X})
        self.assertEqualArray(est.predict(X).astype(np.int64), label)
        self.assertEqualArray(est.predict_proba(X).astype(np.float32), proba, atol=1e-5)


@requires_sklearn("1.4")
class TestSklearnOnnxComparisonWithSkl2onnx(ExtTestCase):
    """Check that yobx and skl2onnx produce numerically equivalent predictions.

    Each test converts the same fitted estimator with both converters and
    asserts that the outputs from ONNX Runtime agree.
    """

    def _compare(self, estimator, X, y=None, atol=1e-4):
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
        except ImportError:
            self.skipTest("skl2onnx not installed")

        if y is not None:
            estimator.fit(X, y)
        else:
            estimator.fit(X)

        # yobx conversion
        yobx_onx = to_onnx(estimator, (X,))
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(yobx_onx))

        # skl2onnx conversion; zipmap=False is only valid for classifiers
        dtype = DoubleTensorType if X.dtype == np.float64 else FloatTensorType
        skl2onnx_opts = {"zipmap": False} if hasattr(estimator, "predict_proba") else {}
        skl_onx = convert_sklearn(
            estimator,
            initial_types=[("X", dtype([None, X.shape[1]]))],
            options=skl2onnx_opts,
        )

        import onnxruntime as rt

        def _run(model_proto, x):
            sess = rt.InferenceSession(
                model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            return sess.run(None, {"X": x})

        yobx_out = _run(yobx_onx, X)
        skl_out = _run(skl_onx, X)

        # Compare first output (labels / predictions)
        self.assertEqualArray(yobx_out[0], skl_out[0], atol=atol)

        # Compare second output (probabilities / scores) when present in both
        if len(yobx_out) > 1 and len(skl_out) > 1:
            self.assertEqualArray(yobx_out[1], skl_out[1], atol=atol)

    def test_linear_regression_compare(self):
        from sklearn.linear_model import LinearRegression

        X, y = _make_regression()
        self._compare(LinearRegression(), X, y)

    def test_logistic_regression_binary_compare(self):
        from sklearn.linear_model import LogisticRegression

        X, y = _make_binary()
        self._compare(LogisticRegression(random_state=0), X, y)

    def test_logistic_regression_multiclass_compare(self):
        from sklearn.linear_model import LogisticRegression

        X, y = _make_multiclass()
        self._compare(LogisticRegression(random_state=0, max_iter=300), X, y)

    def test_random_forest_classifier_compare(self):
        from sklearn.ensemble import RandomForestClassifier

        X, y = _make_binary()
        self._compare(RandomForestClassifier(n_estimators=5, random_state=0), X, y)

    def test_random_forest_regressor_compare(self):
        from sklearn.ensemble import RandomForestRegressor

        X, y = _make_regression()
        self._compare(RandomForestRegressor(n_estimators=5, random_state=0), X, y)

    def test_standard_scaler_compare(self):
        from sklearn.preprocessing import StandardScaler

        X, _ = _make_regression()
        self._compare(StandardScaler(), X)

    def test_decision_tree_classifier_compare(self):
        from sklearn.tree import DecisionTreeClassifier

        X, y = _make_multiclass()
        self._compare(DecisionTreeClassifier(random_state=0), X, y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
