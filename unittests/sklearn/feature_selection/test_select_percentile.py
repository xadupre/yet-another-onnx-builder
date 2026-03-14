"""
Unit tests for yobx.sklearn.feature_selection.SelectPercentile converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestSelectPercentile(ExtTestCase):
    def _make_data(self, dtype, n_samples=50, n_features=10, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, n_features)).astype(dtype)
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_select_percentile_float32(self):
        from sklearn.feature_selection import SelectPercentile, f_classif
        from yobx.sklearn import to_onnx

        X, y = self._make_data(np.float32)
        sel = SelectPercentile(score_func=f_classif, percentile=50)
        sel.fit(X, y)

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

    def test_select_percentile_float64(self):
        from sklearn.feature_selection import SelectPercentile, f_classif
        from yobx.sklearn import to_onnx

        X, y = self._make_data(np.float64)
        sel = SelectPercentile(score_func=f_classif, percentile=50)
        sel.fit(X, y)

        onx = to_onnx(sel, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = sel.transform(X)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-10)

    def test_select_percentile_all_features(self):
        from sklearn.feature_selection import SelectPercentile, f_classif
        from yobx.sklearn import to_onnx

        X, y = self._make_data(np.float32, n_features=5)
        sel = SelectPercentile(score_func=f_classif, percentile=100)
        sel.fit(X, y)

        onx = to_onnx(sel, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = sel.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_select_percentile_few_features(self):
        from sklearn.feature_selection import SelectPercentile, f_classif
        from yobx.sklearn import to_onnx

        X, y = self._make_data(np.float32)
        sel = SelectPercentile(score_func=f_classif, percentile=20)
        sel.fit(X, y)

        onx = to_onnx(sel, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = sel.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_select_percentile_pipeline_float32(self):
        from sklearn.feature_selection import SelectPercentile, f_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X, y = self._make_data(np.float32)
        pipe = Pipeline(
            [
                ("sel", SelectPercentile(score_func=f_classif, percentile=50)),
                ("clf", LogisticRegression()),
            ]
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

    def test_select_percentile_pipeline_float64(self):
        from sklearn.feature_selection import SelectPercentile, f_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X, y = self._make_data(np.float64)
        pipe = Pipeline(
            [
                ("sel", SelectPercentile(score_func=f_classif, percentile=50)),
                ("clf", LogisticRegression()),
            ]
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
