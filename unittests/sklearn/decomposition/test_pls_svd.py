"""
Unit tests for yobx.sklearn.decomposition.pls_svd converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestPLSSVD(ExtTestCase):
    def test_pls_svd_float32(self):
        from sklearn.cross_decomposition import PLSSVD
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = rng.standard_normal((20, 3)).astype(np.float32)
        pls = PLSSVD(n_components=2)
        pls.fit(X, y)

        onx = to_onnx(pls, (X,))

        # Check that Sub (centering) and Div (scaling) are present.
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pls.transform(X).astype(np.float32)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_pls_svd_float64(self):
        from sklearn.cross_decomposition import PLSSVD
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 4)).astype(np.float64)
        y = rng.standard_normal((20, 3)).astype(np.float64)
        pls = PLSSVD(n_components=2)
        pls.fit(X, y)

        onx = to_onnx(pls, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pls.transform(X).astype(np.float64)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-10)

    def test_pls_svd_in_pipeline(self):
        from sklearn.cross_decomposition import PLSSVD
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 6)).astype(np.float32)
        y = rng.standard_normal((40, 2)).astype(np.float32)

        pipe = Pipeline([("scaler", StandardScaler()), ("pls", PLSSVD(n_components=2))])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.transform(X).astype(np.float32)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
