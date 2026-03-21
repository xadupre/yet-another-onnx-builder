"""
Unit tests for yobx.sklearn.preprocessing.Normalizer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestNormalizer(ExtTestCase):
    def test_normalizer_l2_default(self):
        from sklearn.preprocessing import Normalizer
        from yobx.sklearn import to_onnx

        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        normalizer = Normalizer()  # default norm='l2'

        onx = to_onnx(normalizer, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("ReduceL2", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = normalizer.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_normalizer_l1(self):
        from sklearn.preprocessing import Normalizer
        from yobx.sklearn import to_onnx

        X = np.array([[1.0, -2.0, 3.0], [0.0, 5.0, -1.0]], dtype=np.float32)
        normalizer = Normalizer(norm="l1")

        onx = to_onnx(normalizer, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("ReduceL1", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = normalizer.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_normalizer_max(self):
        from sklearn.preprocessing import Normalizer
        from yobx.sklearn import to_onnx

        X = np.array([[1.0, 2.0, -4.0], [3.0, -1.0, 2.0]], dtype=np.float32)
        normalizer = Normalizer(norm="max")

        onx = to_onnx(normalizer, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("ReduceMax", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = normalizer.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_normalizer_zero_row(self):
        """Zero-norm rows should be left unchanged (output row = 0)."""
        from sklearn.preprocessing import Normalizer
        from yobx.sklearn import to_onnx

        X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        for norm in ("l1", "l2", "max"):
            normalizer = Normalizer(norm=norm)

            onx = to_onnx(normalizer, (X,))

            ref = ExtendedReferenceEvaluator(onx)
            result = ref.run(None, {"X": X})[0]
            expected = normalizer.transform(X).astype(np.float32)
            self.assertEqualArray(expected, result, atol=1e-5, msg=f"norm={norm}")

            sess = self.check_ort(onx)
            ort_result = sess.run(None, {"X": X})[0]
            self.assertEqualArray(expected, ort_result, atol=1e-5, msg=f"norm={norm}")

    def test_normalizer_float64(self):
        from sklearn.preprocessing import Normalizer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 5)).astype(np.float64)
        normalizer = Normalizer(norm="l2")

        onx = to_onnx(normalizer, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = normalizer.transform(X)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-10)

    def test_pipeline_normalizer_logistic_regression(self):
        from sklearn.preprocessing import Normalizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)

        pipe = Pipeline([("norm", Normalizer(norm="l2")), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("ReduceL2", op_types)
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
