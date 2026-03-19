"""
Unit tests for yobx.sklearn.neighbors RadiusNeighbors converters.

Includes tests ported from the sklearn-onnx test suite
(tests/test_sklearn_nearest_neighbour_converter.py in the onnx/sklearn-onnx
repository) to validate that the yobx converters produce outputs that match
scikit-learn for the scenarios covered there.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestRadiusNeighborsClassifier(ExtTestCase):
    def test_rnn_classifier_basic(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Greater", op_types)
        self.assertIn("Not", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_float64(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(10)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_probabilities(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels, probabilities = results[0], results[1]

        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        expected_proba = clf.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_proba, probabilities, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_rnn_classifier_multiclass(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] * 2).astype(np.int64) % 3  # 3 classes

        clf = RadiusNeighborsClassifier(radius=2.0)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

    def test_rnn_classifier_outlier_label(self):
        """Test that outlier_label is applied to points with no neighbours."""
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X_train[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=0.5, outlier_label="most_frequent")
        clf.fit(X_train, y)

        # Include points that are guaranteed to be outliers (very far from training data)
        X_test = np.vstack(
            [
                X_train[:5],
                np.array(
                    [[100.0, 100.0, 100.0, 100.0], [-100.0, -100.0, -100.0, -100.0]],
                    dtype=np.float32,
                ),
            ]
        )

        onx = to_onnx(clf, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        labels = results[0]

        expected_labels = clf.predict(X_test).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_outlier_label_explicit(self):
        """Test explicit numeric outlier_label."""
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X_train = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X_train[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=0.5, outlier_label=1)
        clf.fit(X_train, y)

        X_test = np.array([[100.0, 100.0, 100.0, 100.0]], dtype=np.float32)

        onx = to_onnx(clf, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        labels = results[0]

        expected_labels = clf.predict(X_test).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_pipeline(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", RadiusNeighborsClassifier(radius=2.0))]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = pipe.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_com_microsoft(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        domains = {oi.domain for oi in onx.opset_import}
        self.assertIn("com.microsoft", domains)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_metrics(self):
        """Test various distance metrics."""
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(30)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        metrics = [
            ("euclidean", {}),
            ("cosine", {}),
            ("manhattan", {}),
            ("chebyshev", {}),
            ("minkowski", {"p": 3}),
        ]
        for metric, kwargs in metrics:
            with self.subTest(metric=metric, **kwargs):
                clf = RadiusNeighborsClassifier(radius=2.0, metric=metric, **kwargs)
                clf.fit(X, y)

                onx = to_onnx(clf, (X,))

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                expected_labels = clf.predict(X).astype(np.int64)
                self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_opset_too_low_raises(self):
        """Opset < 13 must raise NotImplementedError."""
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(7)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0)
        clf.fit(X, y)

        with self.assertRaises(NotImplementedError):
            to_onnx(clf, (X,), target_opset=12)

    def test_rnn_classifier_distance_weights(self):
        """weights='distance' should produce correct results."""
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(8)
        X = rng.standard_normal((30, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0, weights="distance")
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, ort_results[0])


@requires_sklearn("1.4")
class TestRadiusNeighborsRegressor(ExtTestCase):
    def test_rnn_regressor_basic(self):
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = X[:, 0] * 2 + 1

        reg = RadiusNeighborsRegressor(radius=2.0)
        reg.fit(X, y)

        onx = to_onnx(reg, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Greater", op_types)
        self.assertIn("Not", op_types)
        self.assertIn("Div", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = reg.predict(X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_rnn_regressor_float64(self):
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(20)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        y = (X[:, 0] * 2 + 1).astype(np.float64)

        reg = RadiusNeighborsRegressor(radius=2.0)
        reg.fit(X, y)

        onx = to_onnx(reg, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = reg.predict(X).astype(np.float64)
        self.assertEqualArray(expected, predictions, atol=1e-10)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-10)

    def test_rnn_regressor_pipeline(self):
        from sklearn.neighbors import RadiusNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = X[:, 0] + X[:, 1]

        pipe = Pipeline(
            [("scaler", StandardScaler()), ("reg", RadiusNeighborsRegressor(radius=2.0))]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = pipe.predict(X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_rnn_regressor_com_microsoft(self):
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = X[:, 0] * 2 + 1

        reg = RadiusNeighborsRegressor(radius=2.0)
        reg.fit(X, y)

        onx = to_onnx(reg, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected = reg.predict(X).astype(np.float32)
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_rnn_regressor_metrics(self):
        """Test various distance metrics."""
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(40)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)

        metrics = [
            ("euclidean", {}),
            ("cosine", {}),
            ("manhattan", {}),
            ("chebyshev", {}),
            ("minkowski", {"p": 3}),
        ]
        for metric, kwargs in metrics:
            with self.subTest(metric=metric, **kwargs):
                reg = RadiusNeighborsRegressor(radius=2.0, metric=metric, **kwargs)
                reg.fit(X, y)

                onx = to_onnx(reg, (X,))

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                expected = reg.predict(X).astype(np.float32)
                self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_rnn_regressor_outlier_nan(self):
        """Points with no neighbours within radius produce NaN predictions."""
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((30, 4)).astype(np.float32)
        y = X_train[:, 0] * 2 + 1

        reg = RadiusNeighborsRegressor(radius=2.0)
        reg.fit(X_train, y)

        # A point far from all training data should produce NaN
        X_outlier = np.array([[100.0, 100.0, 100.0, 100.0]], dtype=np.float32)

        onx = to_onnx(reg, (X_train,))

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_outlier})
        self.assertTrue(np.isnan(ort_results[0][0]))

    def test_rnn_regressor_opset_too_low_raises(self):
        """Opset < 13 must raise NotImplementedError."""
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y = X[:, 0] + 1

        reg = RadiusNeighborsRegressor(radius=2.0)
        reg.fit(X, y)

        with self.assertRaises(NotImplementedError):
            to_onnx(reg, (X,), target_opset=12)

    def test_rnn_regressor_distance_weights(self):
        """weights='distance' should produce correct results."""
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X = rng.standard_normal((30, 3)).astype(np.float32)
        y = (X[:, 0] + 1).astype(np.float32)

        reg = RadiusNeighborsRegressor(radius=2.0, weights="distance")
        reg.fit(X, y)

        onx = to_onnx(reg, (X,))

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected = reg.predict(X).astype(np.float32)
        self.assertEqualArray(expected, ort_results[0], atol=2e-3)


# ---------------------------------------------------------------------------
# Tests ported from sklearn-onnx
# (onnx/sklearn-onnx · tests/test_sklearn_nearest_neighbour_converter.py)
# ---------------------------------------------------------------------------


@requires_sklearn("1.4")
class TestSklearnOnnxRadiusNeighborsRegressor(ExtTestCase):
    """Ported from sklearn-onnx test_model_knn_regressor*_radius tests."""

    def test_regressor_iris_float32(self):
        """Port of test_model_knn_regressor_radius: basic iris float32."""
        from sklearn.datasets import load_iris
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.float32)

        model = RadiusNeighborsRegressor()
        model.fit(X, y)

        onx = to_onnx(model, (X[:1],))

        sess = self.check_ort(onx)
        # Use only points from the training set (radius ensures neighbours exist)
        got = sess.run(None, {"X": X[:5]})[0]
        exp = model.predict(X[:5]).astype(np.float32)
        self.assertEqualArray(exp, got.ravel(), atol=1e-4)

    def test_regressor_iris_float64(self):
        """Port of test_model_knn_regressor_double_radius: float64 with outlier."""
        from sklearn.datasets import load_iris
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        iris = load_iris()
        X = iris.data.astype(np.float64)
        y = iris.target.astype(np.float64)

        model = RadiusNeighborsRegressor(radius=2.0)
        model.fit(X, y)

        onx = to_onnx(model, (X[:1],))

        sess = self.check_ort(onx)
        # Points within training set — should produce valid predictions
        got = sess.run(None, {"X": X[:7]})[0]
        exp = model.predict(X[:7]).astype(np.float64)
        self.assertEqualArray(exp, got.ravel(), atol=1e-10)

    def test_regressor_integer_labels(self):
        """Port of test_model_knn_regressor_yint_radius: integer y values."""
        from sklearn.datasets import make_regression
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        X, y = make_regression(20, n_features=2, random_state=0, n_informative=3)
        y = (y / 100).astype(np.int64)
        X = X.astype(np.float32)

        model = RadiusNeighborsRegressor(radius=2.0)
        model.fit(X, y)

        onx = to_onnx(model, (X[:1],))

        sess = self.check_ort(onx)
        got = sess.run(None, {"X": X[:7]})[0]
        # Cast to float32 because the ONNX graph uses float32 internally
        exp = model.predict(X[:7]).astype(np.float32)
        self.assertEqualArray(exp, got.ravel(), atol=1e-4)

    def test_regressor_multi_output(self):
        """Port of test_model_knn_regressor2_1_radius: n_targets=2."""
        from sklearn.datasets import make_regression
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        X, y = make_regression(20, n_features=2, random_state=0, n_targets=2, n_informative=3)
        y = (y / 100).astype(np.float32)
        X = X.astype(np.float32)

        model = RadiusNeighborsRegressor(algorithm="brute")
        model.fit(X, y)

        onx = to_onnx(model, (X[:1],))

        sess = self.check_ort(onx)
        got = sess.run(None, {"X": X})[0]
        exp = model.predict(X).astype(np.float32)
        self.assertEqualArray(exp, got, atol=1e-4)

    def test_regressor_distance_weights_large_radius(self):
        """Port of test_model_knn_regressor_weights_distance_11_radius.

        Uses a large radius so all training points are always in-radius,
        matching the sklearn-onnx test which uses radius=100.
        """
        from sklearn.datasets import make_regression
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        X, y = make_regression(20, n_features=2, random_state=0, n_informative=3)
        y = y / 100
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        model = RadiusNeighborsRegressor(weights="distance", algorithm="brute", radius=100)
        model.fit(X, y)

        onx = to_onnx(model, (X[:1],))

        sess = self.check_ort(onx)
        got = sess.run(None, {"X": X})[0]
        exp = model.predict(X).astype(np.float32)
        # Use atol=5e-3: float32 euclidean distance has rounding errors for
        # self-predictions (distance should be 0 but may be ~1e-3 in float32),
        # so inverse-distance weighting can be off by a few 1e-3.
        self.assertEqualArray(exp, got.ravel(), atol=5e-3)

    def test_regressor_multi_output_distance_weights(self):
        """Multi-output regressor with distance weighting."""
        from sklearn.datasets import make_regression
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        X, y = make_regression(20, n_features=2, random_state=0, n_targets=2, n_informative=3)
        y = (y / 100).astype(np.float32)
        X = X.astype(np.float32)

        model = RadiusNeighborsRegressor(weights="distance", algorithm="brute", radius=100)
        model.fit(X, y)

        onx = to_onnx(model, (X[:1],))

        sess = self.check_ort(onx)
        got = sess.run(None, {"X": X})[0]
        exp = model.predict(X).astype(np.float32)
        # float32 euclidean distance rounding may cause ~5e-3 errors at self-predictions
        self.assertEqualArray(exp, got, atol=5e-3)


@requires_sklearn("1.4")
class TestSklearnOnnxRadiusNeighborsClassifier(ExtTestCase):
    """Ported from sklearn-onnx test_model_knn_classifier*_radius tests."""

    def test_classifier_binary_iris(self):
        """Port of test_model_knn_classifier_binary_class_radius."""
        from sklearn.datasets import load_iris
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.copy()
        y[y == 2] = 1  # binary: 0 vs 1

        model = RadiusNeighborsClassifier()
        model.fit(X, y)

        onx = to_onnx(model, (X[:1],))

        sess = self.check_ort(onx)
        got = sess.run(None, {"X": X})[0]
        exp = model.predict(X).astype(np.int64)
        self.assertEqualArray(exp, got)

    def test_classifier_multiclass_iris(self):
        """Port of test_model_knn_classifier_multi_class_radius."""
        from sklearn.datasets import load_iris
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target

        model = RadiusNeighborsClassifier()
        model.fit(X, y)

        onx = to_onnx(model, (X[:1],))

        sess = self.check_ort(onx)
        got = sess.run(None, {"X": X[:5]})[0]
        exp = model.predict(X[:5]).astype(np.int64)
        self.assertEqualArray(exp, got)

    def test_classifier_multilabel_distance_weights(self):
        """Port of test_model_knn_iris_classifier_multi_reg2_weight_radius.

        Multi-output binary labels with weights='distance'.
        """
        from sklearn.datasets import load_iris
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int64)
        y = np.vstack([(y + 1) % 2, y % 2]).T  # (150, 2) binary multi-label

        model = RadiusNeighborsClassifier(algorithm="brute", weights="distance")
        model.fit(X[:13], y[:13])

        onx = to_onnx(model, (X[:1],))

        sess = self.check_ort(onx)
        got = sess.run(None, {"X": X[:11]})[0]
        exp = model.predict(X[:11]).astype(np.int64)
        self.assertEqualArray(exp, got)

    def test_classifier_multilabel_uniform_weights(self):
        """Multi-output binary labels with uniform weights."""
        from sklearn.datasets import load_iris
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int64)
        y = np.vstack([(y + 1) % 2, y % 2]).T  # (150, 2) binary multi-label

        model = RadiusNeighborsClassifier(algorithm="brute", weights="uniform")
        model.fit(X[:13], y[:13])

        onx = to_onnx(model, (X[:1],))

        sess = self.check_ort(onx)
        got = sess.run(None, {"X": X[:11]})[0]
        exp = model.predict(X[:11]).astype(np.int64)
        self.assertEqualArray(exp, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
