"""
Unit tests for yobx.sklearn.feature_extraction.FeatureHasher converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn


@requires_sklearn("1.4")
class TestFeatureHasherErrors(ExtTestCase):
    """Tests that the converter raises NotImplementedError for unsupported usage."""

    def test_raises_without_com_microsoft_opset(self):
        """Converting without com.microsoft opset raises NotImplementedError."""
        import onnx
        from onnx.helper import make_tensor_value_info
        from yobx.sklearn import to_onnx
        from sklearn.feature_extraction import FeatureHasher

        fh = FeatureHasher(n_features=10, dtype=np.float32)
        vi_names = make_tensor_value_info("X", onnx.TensorProto.STRING, [None, 2])
        vi_vals = make_tensor_value_info("X_values", onnx.TensorProto.FLOAT, [None, 2])

        with self.assertRaises(NotImplementedError):
            to_onnx(fh, (vi_names, vi_vals), target_opset={"": 18})

    def test_raises_with_float_input_no_com_microsoft(self):
        """Passing a float matrix without com.microsoft raises NotImplementedError."""
        from yobx.sklearn import to_onnx
        from sklearn.feature_extraction import FeatureHasher

        fh = FeatureHasher(n_features=10, dtype=np.float32)
        X = fh.transform([{"dog": 1.0, "cat": 2.0}]).toarray().astype(np.float32)

        with self.assertRaises(NotImplementedError):
            to_onnx(fh, (X,))

    def test_raises_with_float_input_with_com_microsoft(self):
        """Passing a float matrix even with com.microsoft opset raises NotImplementedError."""
        from yobx.sklearn import to_onnx
        from sklearn.feature_extraction import FeatureHasher

        fh = FeatureHasher(n_features=10, dtype=np.float32)
        X = fh.transform([{"dog": 1.0, "cat": 2.0}]).toarray().astype(np.float32)

        with self.assertRaises(NotImplementedError):
            to_onnx(fh, (X,), target_opset={"": 18, "com.microsoft": 1})


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

        _N, K = X_names.shape
        target_dtype = np.dtype(fh.dtype)

        vi_names = make_tensor_value_info("X", onnx.TensorProto.STRING, [None, K])
        vi_vals = make_tensor_value_info(
            "X_values",
            onnx.TensorProto.FLOAT if target_dtype == np.float32 else onnx.TensorProto.DOUBLE,
            [None, K],
        )
        return to_onnx(fh, (vi_names, vi_vals), target_opset={"": 18, "com.microsoft": 1})

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
        op_types = [(n.op_type, n.domain) for n in onx.proto.graph.node]
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

        expected = fh.transform([["foo", "bar"], ["baz"]]).toarray().astype(np.float32)
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
