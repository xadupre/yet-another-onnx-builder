"""
Unit tests for yobx.sklearn.feature_extraction.FeatureHasher converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestFeatureHasherPreHashed(ExtTestCase):
    """Tests for the fallback pre-hashed float-matrix path."""

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


@requires_sklearn("1.4")
class TestFeatureHasherMurmurHash(ExtTestCase):
    """Tests for the com.microsoft.MurmurHash3 native path."""

    def _to_onnx_murmurhash(self, fh, X_names, X_values):
        """Convert FeatureHasher with the MurmurHash3 path.

        Passes a STRING tensor as the primary input and a float tensor as the
        second input. Both are passed as ValueInfoProto descriptors.
        """
        import onnx
        from onnx.helper import make_tensor_value_info
        from yobx.sklearn import to_onnx

        N, K = X_names.shape
        target_dtype = np.dtype(fh.dtype)

        vi_names = make_tensor_value_info("X", onnx.TensorProto.STRING, [None, K])
        vi_vals = make_tensor_value_info(
            "X_values",
            onnx.TensorProto.FLOAT
            if target_dtype == np.float32
            else onnx.TensorProto.DOUBLE,
            [None, K],
        )
        return to_onnx(
            fh,
            (vi_names, vi_vals),
            target_opset={"": 18, "com.microsoft": 1},
        )

    def test_murmurhash_string_input_float32(self):
        """Native path: 'string' input_type, float32, alternate_sign=True."""
        from sklearn.feature_extraction import FeatureHasher

        n_features = 10
        fh = FeatureHasher(n_features=n_features, input_type="string", dtype=np.float32)

        X_raw = [["dog", "cat"], ["fish", ""]]
        X_names = np.array(X_raw)
        X_values = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        onx = self._to_onnx_murmurhash(fh, X_names, X_values)

        # MurmurHash3 must be in the graph
        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("MurmurHash3", "com.microsoft"), op_types)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X_names, "X_values": X_values})[0]

        expected = fh.transform([["dog", "cat"], ["fish"]]).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_murmurhash_dict_input_float32(self):
        """Native path: 'dict' input_type, float32."""
        from sklearn.feature_extraction import FeatureHasher

        n_features = 16
        fh = FeatureHasher(n_features=n_features, input_type="dict", dtype=np.float32)

        X_raw = [{"dog": 1.0, "cat": 2.0}, {"fish": 3.0, "dog": 0.5}]
        # Pad to same width
        X_names = np.array([["dog", "cat"], ["fish", "dog"]])
        X_values = np.array([[1.0, 2.0], [3.0, 0.5]], dtype=np.float32)

        onx = self._to_onnx_murmurhash(fh, X_names, X_values)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X_names, "X_values": X_values})[0]

        expected = fh.transform(X_raw).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_murmurhash_no_alternate_sign(self):
        """Native path with alternate_sign=False."""
        from sklearn.feature_extraction import FeatureHasher

        n_features = 8
        fh = FeatureHasher(
            n_features=n_features, input_type="string", dtype=np.float32, alternate_sign=False
        )

        X_names = np.array([["foo", "bar"], ["baz", ""]])
        X_values = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        onx = self._to_onnx_murmurhash(fh, X_names, X_values)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X_names, "X_values": X_values})[0]

        expected = (
            fh.transform([["foo", "bar"], ["baz"]]).toarray().astype(np.float32)
        )
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_murmurhash_output_shape(self):
        """Output shape is (N, n_features) for native MurmurHash3 path."""
        from sklearn.feature_extraction import FeatureHasher

        n_features = 64
        fh = FeatureHasher(n_features=n_features, input_type="string", dtype=np.float32)

        X_names = np.array([["a", "b", "c"], ["d", "e", ""]])
        X_values = np.ones((2, 3), dtype=np.float32)
        X_values[1, 2] = 0.0  # padding

        onx = self._to_onnx_murmurhash(fh, X_names, X_values)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X_names, "X_values": X_values})[0]

        self.assertEqual(result.shape, (2, n_features))


if __name__ == "__main__":
    unittest.main(verbosity=2)
