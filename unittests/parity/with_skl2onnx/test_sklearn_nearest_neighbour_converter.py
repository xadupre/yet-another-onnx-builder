"""
Parity tests for radius nearest-neighbour converters.

These tests are ported from the sklearn-onnx test suite
(onnx/sklearn-onnx · tests/test_sklearn_nearest_neighbour_converter.py)
and verify that the yobx converters produce results matching scikit-learn for
the scenarios covered there.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn


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
