"""
Unit tests for yobx.sklearn.neighbors NeighborhoodComponentsAnalysis converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.0")
class TestNeighborhoodComponentsAnalysis(ExtTestCase):
    def test_nca_basic_float32(self):
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=0)
        nca.fit(X, y)

        onx = to_onnx(nca, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        output = results[0]

        expected = nca.transform(X).astype(np.float32)
        self.assertEqualArray(expected, output, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_nca_basic_float64(self):
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 4)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.int64)

        nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=0)
        nca.fit(X, y)

        onx = to_onnx(nca, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        output = results[0]

        expected = nca.transform(X)
        self.assertEqualArray(expected, output, atol=1e-10)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-10)

    def test_nca_full_components(self):
        """Test NCA with n_components == n_features (identity-like)."""
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        nca = NeighborhoodComponentsAnalysis(n_components=3, random_state=0)
        nca.fit(X, y)

        onx = to_onnx(nca, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        output = results[0]

        expected = nca.transform(X).astype(np.float32)
        self.assertEqualArray(expected, output, atol=1e-5)

    def test_nca_multiclass(self):
        """Test NCA with multiclass labels."""
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((60, 6)).astype(np.float32)
        y = (X[:, 0] * 3).astype(np.int64) % 3  # 3 classes

        nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=0)
        nca.fit(X, y)

        onx = to_onnx(nca, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        output = results[0]

        expected = nca.transform(X).astype(np.float32)
        self.assertEqualArray(expected, output, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
