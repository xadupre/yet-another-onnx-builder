"""
Unit tests for yobx.sklearn.feature_extraction.FeatureHasher converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestFeatureHasher(ExtTestCase):
    def _make_hashed(self, n_features=10, dtype=np.float32):
        """Return a pre-hashed dense matrix for two samples."""
        from sklearn.feature_extraction import FeatureHasher

        fh = FeatureHasher(n_features=n_features, dtype=dtype)
        X_raw = [{"dog": 1.0, "cat": 2.0}, {"fish": 3.0, "cat": -0.5}]
        return fh, fh.transform(X_raw).toarray().astype(dtype)

    def test_feature_hasher_float32(self):
        from sklearn.feature_extraction import FeatureHasher
        from yobx.sklearn import to_onnx

        fh, X = self._make_hashed(dtype=np.float32)

        onx = to_onnx(fh, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Identity", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(X, result, atol=1e-7)

    def test_feature_hasher_float64(self):
        from sklearn.feature_extraction import FeatureHasher
        from yobx.sklearn import to_onnx

        fh, X = self._make_hashed(dtype=np.float64)

        onx = to_onnx(fh, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(X, result, atol=1e-10)

    def test_feature_hasher_cast_dtype(self):
        """Input dtype differs from estimator dtype → Cast node emitted."""
        from sklearn.feature_extraction import FeatureHasher
        from yobx.sklearn import to_onnx

        fh = FeatureHasher(n_features=10, dtype=np.float64)
        _, X64 = self._make_hashed(n_features=10, dtype=np.float64)
        # Provide float32 input; converter must cast to float64.
        X32 = X64.astype(np.float32)

        onx = to_onnx(fh, (X32,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Cast", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X32})[0]
        self.assertEqual(result.dtype, np.float64)
        self.assertEqualArray(X64, result, atol=1e-5)

    def test_feature_hasher_output_shape(self):
        """Output shape must be (n_samples, n_features)."""
        from sklearn.feature_extraction import FeatureHasher
        from yobx.sklearn import to_onnx

        n_features = 32
        fh = FeatureHasher(n_features=n_features, dtype=np.float32)
        X_raw = [{"a": 1.0, "b": 2.0, "c": -1.0}, {"d": 3.0}]
        X = fh.transform(X_raw).toarray().astype(np.float32)

        onx = to_onnx(fh, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]

        self.assertEqual(result.shape, (2, n_features))

    def test_feature_hasher_no_alternate_sign(self):
        """alternate_sign=False – values are stored without sign flip."""
        from sklearn.feature_extraction import FeatureHasher
        from yobx.sklearn import to_onnx

        fh = FeatureHasher(n_features=16, dtype=np.float32, alternate_sign=False)
        X_raw = [{"foo": 1.0, "bar": 5.0}]
        X = fh.transform(X_raw).toarray().astype(np.float32)

        onx = to_onnx(fh, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(X, result, atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
