"""
Unit tests for yobx.sklearn.preprocessing.MinMaxScaler converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestMinMaxScaler(ExtTestCase):
    def test_min_max_scaler(self):
        from sklearn.preprocessing import MinMaxScaler
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        mms = MinMaxScaler()
        mms.fit(X)

        onx = to_onnx(mms, (X,))

        # Check graph structure: default (0,1) range uses Sub + Div + Identity
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)

        # Check numerical output
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mms.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_min_max_scaler_feature_range(self):
        from sklearn.preprocessing import MinMaxScaler
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        mms = MinMaxScaler(feature_range=(-1, 1))
        mms.fit(X)

        onx = to_onnx(mms, (X,))

        # Non-default feature_range adds Mul + Add on top of Sub + Div
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("Mul", op_types)
        self.assertIn("Add", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mms.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_pipeline_min_max_scaler_logistic_regression(self):
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("Gemm", op_types)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
