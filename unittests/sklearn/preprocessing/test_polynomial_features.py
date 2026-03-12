"""
Unit tests for yobx.sklearn.preprocessing.PolynomialFeatures converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestPolynomialFeatures(ExtTestCase):
    def test_polynomial_features_degree2(self):
        from sklearn.preprocessing import PolynomialFeatures
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        pf = PolynomialFeatures(degree=2)
        pf.fit(X)

        onx = to_onnx(pf, (X,))

        # Verify graph contains expected ops
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Unsqueeze", op_types)
        self.assertIn("Where", op_types)
        self.assertIn("Pow", op_types)
        self.assertIn("ReduceProd", op_types)

        # Numerical check via reference evaluator
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pf.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        # Check with OnnxRuntime
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_polynomial_features_degree1(self):
        from sklearn.preprocessing import PolynomialFeatures
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        pf = PolynomialFeatures(degree=1)
        pf.fit(X)

        onx = to_onnx(pf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pf.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_polynomial_features_no_bias(self):
        from sklearn.preprocessing import PolynomialFeatures
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        pf = PolynomialFeatures(degree=2, include_bias=False)
        pf.fit(X)

        onx = to_onnx(pf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pf.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_polynomial_features_interaction_only(self):
        from sklearn.preprocessing import PolynomialFeatures
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        pf = PolynomialFeatures(degree=2, interaction_only=True)
        pf.fit(X)

        onx = to_onnx(pf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pf.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_polynomial_features_degree3(self):
        from sklearn.preprocessing import PolynomialFeatures
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [0, 1]], dtype=np.float32)
        pf = PolynomialFeatures(degree=3)
        pf.fit(X)

        onx = to_onnx(pf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pf.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_polynomial_features_with_zeros(self):
        """Test that 0^0 is handled correctly (should equal 1)."""
        from sklearn.preprocessing import PolynomialFeatures
        from yobx.sklearn import to_onnx

        X = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.float32)
        pf = PolynomialFeatures(degree=2)
        pf.fit(X)

        onx = to_onnx(pf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pf.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_pipeline_polynomial_features_linear_regression(self):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        pipe = Pipeline(
            [("poly", PolynomialFeatures(degree=2)), ("lr", LinearRegression())]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.predict(X).astype(np.float32)
        self.assertEqualArray(expected, result.ravel(), atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result.ravel(), atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
