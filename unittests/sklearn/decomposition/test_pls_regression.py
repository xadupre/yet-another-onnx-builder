"""
Unit tests for yobx.sklearn.decomposition.pls_regression converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestPLSRegression(ExtTestCase):
    def test_pls_regression_single_target(self):
        from sklearn.cross_decomposition import PLSRegression
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = rng.standard_normal(20).astype(np.float32)
        pls = PLSRegression(n_components=2)
        pls.fit(X, y)

        onx = to_onnx(pls, (X,))

        # Check that Sub (centering) is present and a linear transform (MatMul or
        # fused Gemm) is present.
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertTrue(
            "MatMul" in op_types or "Gemm" in op_types,
            f"Expected MatMul or Gemm in {op_types}",
        )

        # Check numerical output.
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pls.predict(X).astype(np.float32)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_pls_regression_multi_target(self):
        from sklearn.cross_decomposition import PLSRegression
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 5)).astype(np.float32)
        y = rng.standard_normal((30, 3)).astype(np.float32)
        pls = PLSRegression(n_components=3)
        pls.fit(X, y)

        onx = to_onnx(pls, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pls.predict(X).astype(np.float32)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_pls_regression_in_pipeline(self):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 6)).astype(np.float32)
        y = rng.standard_normal((40, 2)).astype(np.float32)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=3)),
        ])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.predict(X).astype(np.float32)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
