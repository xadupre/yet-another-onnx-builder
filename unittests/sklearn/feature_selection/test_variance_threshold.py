"""
Unit tests for yobx.sklearn.feature_selection.VarianceThreshold converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestVarianceThreshold(ExtTestCase):
    def _make_data(self, dtype, n_samples=50, n_features=10, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, n_features)).astype(dtype)
        # Make the first column nearly constant so it gets filtered out
        X[:, 0] = 0.01 * rng.standard_normal(n_samples).astype(dtype)
        return X

    def test_variance_threshold_float32(self):
        from sklearn.feature_selection import VarianceThreshold
        from yobx.sklearn import to_onnx

        X = self._make_data(np.float32)
        sel = VarianceThreshold(threshold=0.1)
        sel.fit(X)

        onx = to_onnx(sel, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gather", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = sel.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_variance_threshold_float64(self):
        from sklearn.feature_selection import VarianceThreshold
        from yobx.sklearn import to_onnx

        X = self._make_data(np.float64)
        sel = VarianceThreshold(threshold=0.1)
        sel.fit(X)

        onx = to_onnx(sel, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = sel.transform(X)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-10)

    def test_variance_threshold_zero_threshold(self):
        from sklearn.feature_selection import VarianceThreshold
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 6)).astype(np.float32)
        # Add a completely constant column
        X[:, 2] = 5.0
        sel = VarianceThreshold()
        sel.fit(X)

        onx = to_onnx(sel, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = sel.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_variance_threshold_all_features(self):
        from sklearn.feature_selection import VarianceThreshold
        from yobx.sklearn import to_onnx

        X = self._make_data(np.float32)
        sel = VarianceThreshold(threshold=0.0)
        sel.fit(X)

        onx = to_onnx(sel, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = sel.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_variance_threshold_pipeline_float32(self):
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = self._make_data(np.float32)
        y = (X[:, 1] > 0).astype(int)
        pipe = Pipeline(
            [("sel", VarianceThreshold(threshold=0.1)), ("clf", LogisticRegression())]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gather", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_variance_threshold_pipeline_float64(self):
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = self._make_data(np.float64)
        y = (X[:, 1] > 0).astype(int)
        pipe = Pipeline(
            [("sel", VarianceThreshold(threshold=0.1)), ("clf", LogisticRegression())]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-9)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
