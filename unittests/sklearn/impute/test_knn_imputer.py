"""
Unit tests for yobx.sklearn.preprocessing.KNNImputer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestKNNImputer(ExtTestCase):
    def test_knn_imputer_uniform_basic(self):
        """Basic NaN imputation with uniform weights."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]], dtype=np.float32)
        imp = KNNImputer(n_neighbors=2, weights="uniform")
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = imp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_knn_imputer_distance_weights(self):
        """NaN imputation with distance weights."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]], dtype=np.float32)
        imp = KNNImputer(n_neighbors=2, weights="distance")
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = imp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_knn_imputer_float64(self):
        """float64 input is handled correctly."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 3.0], [np.nan, 6.0, 5.0]], dtype=np.float64)
        imp = KNNImputer(n_neighbors=2)
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = imp.transform(X)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_knn_imputer_no_nans(self):
        """If no values are missing, the output equals the input."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        X_test = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        imp = KNNImputer(n_neighbors=2)
        imp.fit(X_train)

        onx = to_onnx(imp, (X_test,))

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        expected = imp.transform(X_test).astype(np.float32)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_knn_imputer_multiple_nans_per_row(self):
        """Multiple NaN values in the same row are all imputed."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(42)
        X = rng.random((10, 4)).astype(np.float32)
        X[0, 1] = np.nan
        X[0, 3] = np.nan
        X[3, 0] = np.nan
        X[7, 2] = np.nan

        imp = KNNImputer(n_neighbors=3)
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        expected = imp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_knn_imputer_k1(self):
        """n_neighbors=1: use nearest-neighbour value directly."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 3.0], [5.0, 6.0, 5.0]], dtype=np.float32)
        imp = KNNImputer(n_neighbors=1)
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = imp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_knn_imputer_larger_k(self):
        """Larger k with distance weights on random data."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.random((20, 5)).astype(np.float32)
        # Randomly insert NaN values.
        nan_mask = rng.random((20, 5)) < 0.2
        X[nan_mask] = np.nan

        imp = KNNImputer(n_neighbors=5, weights="distance")
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        expected = imp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, ort_result, atol=1e-3)

    def test_knn_imputer_cdist_uniform(self):
        """CDist path: uniform weights, float32."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]], dtype=np.float32)
        imp = KNNImputer(n_neighbors=2, weights="uniform")
        imp.fit(X)

        onx = to_onnx(imp, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        expected = imp.transform(X).astype(np.float32)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_knn_imputer_cdist_distance_weights(self):
        """CDist path: distance weights, float32."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]], dtype=np.float32)
        imp = KNNImputer(n_neighbors=2, weights="distance")
        imp.fit(X)

        onx = to_onnx(imp, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        expected = imp.transform(X).astype(np.float32)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_knn_imputer_cdist_float64(self):
        """CDist path: float64 input."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(7)
        X = rng.random((15, 4)).astype(np.float64)
        X[0, 1] = np.nan
        X[3, 2] = np.nan
        X[8, 0] = np.nan

        imp = KNNImputer(n_neighbors=3)
        imp.fit(X)

        onx = to_onnx(imp, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        expected = imp.transform(X)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_knn_imputer_add_indicator_raises(self):
        """add_indicator=True is not supported."""
        from sklearn.impute import KNNImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [np.nan, 3]], dtype=np.float32)
        imp = KNNImputer(add_indicator=True)
        imp.fit(X)

        with self.assertRaises(NotImplementedError):
            to_onnx(imp, (X,))

    def test_knn_imputer_pipeline(self):
        """KNNImputer inside a pipeline followed by a classifier."""
        from sklearn.impute import KNNImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [np.nan, 3], [7, 6], [4, np.nan]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline([("imp", KNNImputer(n_neighbors=2)), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
