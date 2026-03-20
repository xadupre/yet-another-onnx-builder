"""
Unit tests for yobx.sklearn.decomposition.IncrementalPCA converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestIncrementalPCA(ExtTestCase):
    def test_incremental_pca_basic(self):
        from sklearn.decomposition import IncrementalPCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        ipca = IncrementalPCA(n_components=2)
        ipca.fit(X)

        onx = to_onnx(ipca, (X,))

        # Check that Sub (centering) and MatMul (projection) are present.
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)

        # Check numerical output.
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ipca.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_incremental_pca_all_components(self):
        from sklearn.decomposition import IncrementalPCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((15, 3)).astype(np.float32)
        ipca = IncrementalPCA()
        ipca.fit(X)

        onx = to_onnx(ipca, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ipca.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_incremental_pca_whiten(self):
        from sklearn.decomposition import IncrementalPCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        ipca = IncrementalPCA(n_components=2, whiten=True)
        ipca.fit(X)

        onx = to_onnx(ipca, (X,))

        # Whitening adds a Div node.
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)
        self.assertIn("Div", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ipca.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_incremental_pca_partial_fit(self):
        from sklearn.decomposition import IncrementalPCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 5)).astype(np.float32)
        ipca = IncrementalPCA(n_components=3)
        # Train incrementally in two batches.
        ipca.partial_fit(X[:15])
        ipca.partial_fit(X[15:])

        onx = to_onnx(ipca, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ipca.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
