"""
Unit tests for yobx.sklearn.manifold.LocallyLinearEmbedding converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestLocallyLinearEmbedding(ExtTestCase):
    def test_lle_basic(self):
        from sklearn.manifold import LocallyLinearEmbedding
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=5)
        lle.fit(X)

        onx = to_onnx(lle, (X,))

        # Verify key ops are present
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TopK", op_types)
        self.assertIn("MatMul", op_types)

        # Check numerical output with reference evaluator
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = lle.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-3)

        # Check with ONNX Runtime
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-3)

    def test_lle_out_of_sample(self):
        """Test transform on unseen data (the primary use-case for ONNX export)."""
        from sklearn.manifold import LocallyLinearEmbedding
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((50, 5)).astype(np.float32)
        X_test = rng.standard_normal((10, 5)).astype(np.float32)

        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=7)
        lle.fit(X_train)

        onx = to_onnx(lle, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        expected = lle.transform(X_test).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-3)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-3)

    def test_lle_single_component(self):
        from sklearn.manifold import LocallyLinearEmbedding
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((25, 5)).astype(np.float32)
        lle = LocallyLinearEmbedding(n_components=1, n_neighbors=4)
        lle.fit(X)

        onx = to_onnx(lle, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = lle.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-3)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-3)

    def test_lle_float64(self):
        """Test that float64 inputs are handled correctly."""
        from sklearn.manifold import LocallyLinearEmbedding
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=5)
        lle.fit(X)

        onx = to_onnx(lle, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = lle.transform(X)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_lle_more_neighbors(self):
        """Test with a larger number of neighbours."""
        from sklearn.manifold import LocallyLinearEmbedding
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X = rng.standard_normal((40, 6)).astype(np.float32)
        lle = LocallyLinearEmbedding(n_components=3, n_neighbors=10)
        lle.fit(X)

        onx = to_onnx(lle, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = lle.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-3)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
