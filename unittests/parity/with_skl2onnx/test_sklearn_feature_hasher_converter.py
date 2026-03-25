"""
Parity tests for the FeatureHasher converter, mirroring sklearn-onnx's
``test_sklearn_feature_hasher_converter.py``.

All tests compare yobx ONNX results (run via OnnxRuntime) against sklearn's
own ``transform()`` output.  The converter requires the ``com.microsoft``
ONNX domain for the ``MurmurHash3`` operator.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn

# Opset including com.microsoft for MurmurHash3
_OPSET = {"": 18, "com.microsoft": 1}

# Sample documents from sklearn-onnx's FeatureHasher test suite
_CORPUS = [
    "cat dog bird",
    "cat cat fish",
    "dog bird fish bird",
    "ant bee",
]


def _tokenize(docs):
    """Split each string document into whitespace-separated tokens."""
    return [doc.split() for doc in docs]


def _pad_tokens(token_lists):
    """Convert a list of token lists into a padded 2-D string array."""
    if not token_lists or all(len(d) == 0 for d in token_lists):
        return np.array([[""]] * len(token_lists), dtype=object)
    max_len = max((len(d) for d in token_lists), default=1)
    return np.array(
        [list(d) + [""] * (max_len - len(d)) for d in token_lists], dtype=object
    )


@requires_sklearn("1.4")
class TestSklearnFeatureHasherConverter(ExtTestCase):
    """Parity tests for the FeatureHasher ONNX converter.

    Results from the yobx-produced ONNX model (executed with OnnxRuntime)
    are compared against sklearn's ``transform()`` output.  The input arrays
    are padded 2-D string arrays where shorter rows are padded with ``""``.

    Note: the ``com.microsoft`` ONNX domain is required.  Tests are skipped
    automatically on builds where OnnxRuntime does not provide that domain.
    """

    def _to_onnx(self, estimator, X):
        from yobx.sklearn import to_onnx

        return to_onnx(estimator, (X,), target_opset=_OPSET)

    # ------------------------------------------------------------------
    # Mirrors test_model_feature_hasher_10 / test_model_feature_hasher_n_features
    # ------------------------------------------------------------------

    def test_model_feature_hasher_n10(self):
        """n_features=10, alternate_sign=True on the standard corpus."""
        from sklearn.feature_extraction import FeatureHasher

        token_lists = _tokenize(_CORPUS)
        fh = FeatureHasher(n_features=10, input_type="string")
        fh.fit([])

        X = _pad_tokens(token_lists)
        onx = self._to_onnx(fh, X)
        self.assertIsNotNone(onx)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(token_lists).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_model_feature_hasher_n10_no_alt_sign(self):
        """n_features=10, alternate_sign=False on the standard corpus."""
        from sklearn.feature_extraction import FeatureHasher

        token_lists = _tokenize(_CORPUS)
        fh = FeatureHasher(n_features=10, input_type="string", alternate_sign=False)
        fh.fit([])

        X = _pad_tokens(token_lists)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(token_lists).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_model_feature_hasher_large_n_features(self):
        """n_features=1024 matches sklearn on the standard corpus."""
        from sklearn.feature_extraction import FeatureHasher

        token_lists = _tokenize(_CORPUS)
        fh = FeatureHasher(n_features=1024, input_type="string")
        fh.fit([])

        X = _pad_tokens(token_lists)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(token_lists).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    # ------------------------------------------------------------------
    # Mirrors test_model_feature_hasher_float32
    # ------------------------------------------------------------------

    def test_model_feature_hasher_float32(self):
        """dtype=float32 — output matches sklearn's float32-cast result."""
        from sklearn.feature_extraction import FeatureHasher

        token_lists = _tokenize(_CORPUS)
        fh = FeatureHasher(n_features=10, input_type="string", dtype=np.float32)
        fh.fit([])

        X = _pad_tokens(token_lists)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(token_lists).toarray().astype(np.float32)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

    # ------------------------------------------------------------------
    # Mirrors test_model_feature_hasher_more_tokens
    # ------------------------------------------------------------------

    def test_model_feature_hasher_more_tokens(self):
        """Longer token lists — more padding needed."""
        from sklearn.feature_extraction import FeatureHasher

        token_lists = [
            ["one", "two", "three", "four", "five", "six"],
            ["alpha", "beta"],
            ["x", "y", "z", "x", "x"],
            [],
        ]
        fh = FeatureHasher(n_features=16, input_type="string")
        fh.fit([])

        X = _pad_tokens(token_lists)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(token_lists).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    # ------------------------------------------------------------------
    # Mirrors test_model_feature_hasher_collision
    # ------------------------------------------------------------------

    def test_model_feature_hasher_collision(self):
        """Forced collisions with n_features=2 accumulate correctly."""
        from sklearn.feature_extraction import FeatureHasher

        token_lists = [
            ["apple", "banana", "cherry", "date"],
            ["fig", "grape", "honeydew"],
        ]
        fh = FeatureHasher(n_features=2, input_type="string", alternate_sign=True)
        fh.fit([])

        X = _pad_tokens(token_lists)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(token_lists).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    # ------------------------------------------------------------------
    # Mirrors test_model_feature_hasher_empty_strings
    # ------------------------------------------------------------------

    def test_model_feature_hasher_empty_doc(self):
        """An empty document produces an all-zero feature vector."""
        from sklearn.feature_extraction import FeatureHasher

        token_lists = [["cat", "dog"], [], ["bird"]]
        fh = FeatureHasher(n_features=8, input_type="string")
        fh.fit([])

        X = _pad_tokens(token_lists)
        onx = self._to_onnx(fh, X)

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        expected = fh.transform(token_lists).toarray()
        self.assertEqualArray(expected, result, atol=1e-6)

    # ------------------------------------------------------------------
    # Pipeline test (mirrors test_feature_hasher_pipeline)
    # ------------------------------------------------------------------

    def test_model_feature_hasher_in_pipeline(self):
        """FeatureHasher inside a Pipeline converts correctly."""
        from sklearn.feature_extraction import FeatureHasher
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        token_lists = _tokenize(_CORPUS)
        X = _pad_tokens(token_lists)
        y = np.array([0, 1, 0, 1])

        fh = FeatureHasher(n_features=16, input_type="string", dtype=np.float32)
        clf = LogisticRegression(max_iter=200, random_state=0)
        pipe = Pipeline([("fh", fh), ("clf", clf)])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,), target_opset=_OPSET)
        self.assertIsNotNone(onx)

        sess = self.check_ort(onx)
        result_labels = sess.run(None, {"X": X})[0]
        expected_labels = pipe.predict(X)
        self.assertEqualArray(expected_labels, result_labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
