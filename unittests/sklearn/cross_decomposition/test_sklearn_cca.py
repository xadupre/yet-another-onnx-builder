"""
Unit tests for yobx.sklearn.cross_decomposition.cca converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestSklearnCCA(ExtTestCase):
    def test_cca_float32(self):
        from sklearn.cross_decomposition import CCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        Y = rng.standard_normal((50, 3)).astype(np.float32)
        cca = CCA(n_components=2)
        cca.fit(X, Y)

        onx = to_onnx(cca, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cca.transform(X).astype(np.float32)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_cca_float64(self):
        from sklearn.cross_decomposition import CCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 4)).astype(np.float64)
        Y = rng.standard_normal((50, 3)).astype(np.float64)
        cca = CCA(n_components=2)
        cca.fit(X, Y)

        onx = to_onnx(cca, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cca.transform(X).astype(np.float64)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-10)

    def test_cca_in_pipeline(self):
        from sklearn.cross_decomposition import CCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 5)).astype(np.float32)
        Y = rng.standard_normal((60, 3)).astype(np.float32)

        pipe = Pipeline([("scaler", StandardScaler()), ("cca", CCA(n_components=2))])
        pipe.fit(X, Y)

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
