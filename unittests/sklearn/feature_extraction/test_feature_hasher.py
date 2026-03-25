"""
Unit tests for yobx.sklearn.feature_extraction.FeatureHasher converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn


def _pad_tokens(docs):
    """Convert a list of token lists into a padded 2-D string array."""
    if not docs or all(len(d) == 0 for d in docs):
        return np.array([[""]] * len(docs), dtype=object)
    max_len = max((len(d) for d in docs), default=1)
    max_len = max(max_len, 1)
    return np.array([list(d) + [""] * (max_len - len(d)) for d in docs], dtype=object)


@requires_sklearn("1.4")
class TestFeatureHasher(ExtTestCase):
    """Tests for the FeatureHasher → ONNX converter.

    The converter requires the ``com.microsoft`` ONNX domain (for
    MurmurHash3) and only supports ``input_type='string'``.

    Empty strings ``""`` in the input array are treated as padding and
    contribute nothing to the output feature vector.
    """

    _OPSET = {"": 18, "com.microsoft": 1}

    def _to_onnx(self, estimator, X):
        from yobx.sklearn import to_onnx

        return to_onnx(estimator, (X,), target_opset=self._OPSET)

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_basic_alternate_sign_true(self):
        """alternate_sign=True: signs match sklearn's murmurhash3_32 sign."""
        from sklearn.feature_extraction import FeatureHasher

        docs = [["foo", "bar", "foo"], ["baz"], []]
        fh = FeatureHasher(n_features=16, input_type="string", alternate_sign=True)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        op_types = [(n.op_type, n.domain) for n in onx.proto.graph.node]
        self.assertIn(("MurmurHash3", "com.microsoft"), op_types)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(docs).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_basic_alternate_sign_false(self):
        """alternate_sign=False: all contributions are +1."""
        from sklearn.feature_extraction import FeatureHasher

        docs = [["foo", "bar", "foo"], ["baz"]]
        fh = FeatureHasher(n_features=16, input_type="string", alternate_sign=False)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(docs).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_float32_dtype(self):
        """dtype=np.float32 produces a FLOAT output tensor."""
        from sklearn.feature_extraction import FeatureHasher

        docs = [["hello", "world"], ["test"]]
        fh = FeatureHasher(n_features=8, input_type="string", dtype=np.float32)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        # OnnxRuntime output dtype should be float32
        self.assertEqual(result.dtype, np.float32)
        expected = fh.transform(docs).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_float64_dtype(self):
        """dtype=np.float64 (the sklearn default) produces a DOUBLE output tensor."""
        from sklearn.feature_extraction import FeatureHasher

        docs = [["hello", "world"], ["test"]]
        fh = FeatureHasher(n_features=8, input_type="string", dtype=np.float64)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        # OnnxRuntime output dtype should be float64
        self.assertEqual(result.dtype, np.float64)
        expected = fh.transform(docs).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    # ------------------------------------------------------------------
    # Collision and count tests
    # ------------------------------------------------------------------

    def test_repeated_tokens_count(self):
        """Repeated tokens in a document are summed, not deduplicated."""
        from sklearn.feature_extraction import FeatureHasher

        docs = [["a", "a", "a", "b"]]
        fh = FeatureHasher(n_features=32, input_type="string", alternate_sign=False)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(docs).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_collision_two_tokens_same_bucket(self):
        """Tokens that hash to the same bucket accumulate correctly."""
        from sklearn.feature_extraction import FeatureHasher

        # Use n_features=2 to force many collisions
        docs = [["alpha", "beta", "gamma", "delta", "epsilon"]]
        fh = FeatureHasher(n_features=2, input_type="string", alternate_sign=True)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(docs).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    # ------------------------------------------------------------------
    # Empty documents and padding
    # ------------------------------------------------------------------

    def test_empty_document(self):
        """A document with no features produces an all-zero feature vector."""
        from sklearn.feature_extraction import FeatureHasher

        docs = [["foo"], [], ["bar"]]
        fh = FeatureHasher(n_features=16, input_type="string", alternate_sign=True)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(docs).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_all_empty_documents(self):
        """All-empty documents produce an all-zero output."""
        from sklearn.feature_extraction import FeatureHasher

        docs = [[], [], []]
        fh = FeatureHasher(n_features=8, input_type="string", alternate_sign=True)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(docs).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_single_sample(self):
        """Single-sample batch works correctly."""
        from sklearn.feature_extraction import FeatureHasher

        docs = [["hello", "world", "hello"]]
        fh = FeatureHasher(n_features=16, input_type="string", alternate_sign=True)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(docs).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    # ------------------------------------------------------------------
    # n_features variations
    # ------------------------------------------------------------------

    def test_n_features_power_of_two(self):
        """n_features=1024 (power of 2) works correctly."""
        from sklearn.feature_extraction import FeatureHasher

        docs = [["cat", "dog", "bird", "cat", "fish"]]
        fh = FeatureHasher(n_features=1024, input_type="string", alternate_sign=True)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(docs).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_n_features_non_power_of_two(self):
        """n_features=10 (non-power-of-two) produces correct modular hashing."""
        from sklearn.feature_extraction import FeatureHasher

        docs = [["one", "two", "three"], ["four", "five"]]
        fh = FeatureHasher(n_features=10, input_type="string", alternate_sign=True)
        fh.fit([])

        X = _pad_tokens(docs)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(docs).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_raises_for_dict_input_type(self):
        """input_type='dict' must raise NotImplementedError."""
        from sklearn.feature_extraction import FeatureHasher
        from yobx.sklearn import to_onnx

        fh = FeatureHasher(n_features=8, input_type="dict")
        fh.fit([])
        X = np.array([["foo", "bar"]], dtype=object)
        with self.assertRaises(NotImplementedError):
            to_onnx(fh, (X,), target_opset=self._OPSET)

    def test_raises_without_com_microsoft_opset(self):
        """Converter raises RuntimeError when com.microsoft opset is absent."""
        from sklearn.feature_extraction import FeatureHasher
        from yobx.sklearn import to_onnx

        fh = FeatureHasher(n_features=8, input_type="string")
        fh.fit([])
        X = np.array([["foo", "bar"]], dtype=object)
        with self.assertRaises(RuntimeError):
            to_onnx(fh, (X,), target_opset={"": 18})

    def test_raises_for_non_string_input(self):
        """Float input must raise NotImplementedError."""
        from sklearn.feature_extraction import FeatureHasher
        from yobx.sklearn import to_onnx

        fh = FeatureHasher(n_features=8, input_type="string")
        fh.fit([])
        X_float = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaises(NotImplementedError):
            to_onnx(fh, (X_float,), target_opset=self._OPSET)


if __name__ == "__main__":
    unittest.main(verbosity=2)
