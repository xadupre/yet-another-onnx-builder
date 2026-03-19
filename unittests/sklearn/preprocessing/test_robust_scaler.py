"""
Unit tests for yobx.sklearn.preprocessing.RobustScaler converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestRobustScaler(ExtTestCase):
    def test_robust_scaler(self):
        from sklearn.preprocessing import RobustScaler
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        rs = RobustScaler()
        rs.fit(X)

        onx = to_onnx(rs, (X,))

        # Default settings use Sub + Div
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)

        # Check numerical output
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = rs.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_robust_scaler_no_centering(self):
        from sklearn.preprocessing import RobustScaler
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        rs = RobustScaler(with_centering=False)
        rs.fit(X)

        onx = to_onnx(rs, (X,))

        # No centering means no Sub node
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertNotIn("Sub", op_types)
        self.assertIn("Div", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = rs.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_robust_scaler_no_scaling(self):
        from sklearn.preprocessing import RobustScaler
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        rs = RobustScaler(with_scaling=False)
        rs.fit(X)

        onx = to_onnx(rs, (X,))

        # No scaling means no Div node
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertNotIn("Div", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = rs.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_pipeline_robust_scaler_logistic_regression(self):
        from sklearn.preprocessing import RobustScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline([("scaler", RobustScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
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
