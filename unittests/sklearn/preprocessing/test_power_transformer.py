"""
Unit tests for yobx.sklearn.preprocessing.PowerTransformer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestPowerTransformer(ExtTestCase):
    def _make_data(self, seed=42, n=20, n_features=3):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, n_features)).astype(np.float32)

    def test_yeo_johnson_default(self):
        """Default Yeo-Johnson with standardize=True."""
        from sklearn.preprocessing import PowerTransformer
        from yobx.sklearn import to_onnx

        X = self._make_data()
        pt = PowerTransformer()
        pt.fit(X)

        onx = to_onnx(pt, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pt.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_yeo_johnson_no_standardize(self):
        """Yeo-Johnson with standardize=False."""
        from sklearn.preprocessing import PowerTransformer
        from yobx.sklearn import to_onnx

        X = self._make_data()
        pt = PowerTransformer(standardize=False)
        pt.fit(X)

        onx = to_onnx(pt, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pt.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_box_cox(self):
        """Box-Cox transform (requires positive data)."""
        from sklearn.preprocessing import PowerTransformer
        from yobx.sklearn import to_onnx

        X = np.abs(self._make_data()) + 0.1
        pt = PowerTransformer(method="box-cox")
        pt.fit(X)

        onx = to_onnx(pt, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pt.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_box_cox_no_standardize(self):
        """Box-Cox transform with standardize=False."""
        from sklearn.preprocessing import PowerTransformer
        from yobx.sklearn import to_onnx

        X = np.abs(self._make_data()) + 0.1
        pt = PowerTransformer(method="box-cox", standardize=False)
        pt.fit(X)

        onx = to_onnx(pt, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pt.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_pipeline_power_transformer_logistic_regression(self):
        """PowerTransformer inside a Pipeline followed by LogisticRegression."""
        from sklearn.preprocessing import PowerTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = self._make_data(n=40)
        y = (X[:, 0] > 0).astype(np.int64)
        pipe = Pipeline(
            [("pt", PowerTransformer()), ("clf", LogisticRegression())]
        ).fit(X, y)

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
