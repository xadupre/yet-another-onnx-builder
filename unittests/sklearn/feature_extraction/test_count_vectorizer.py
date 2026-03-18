"""
Unit tests for yobx.sklearn.feature_extraction.CountVectorizer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


def _pad_tokens(docs, tokenize=str.split):
    """Tokenise documents and pad rows with '' to a rectangular 2-D array."""
    tokenized = [tokenize(d) for d in docs]
    max_len = max(len(t) for t in tokenized)
    return np.array(
        [t + [""] * (max_len - len(t)) for t in tokenized], dtype=object
    )


@requires_sklearn("1.4")
class TestCountVectorizer(ExtTestCase):
    def test_unigrams_float32(self):
        """Default unigram CountVectorizer on simple documents."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world", "world peace", "hello peace"]
        cv = CountVectorizer()
        cv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(cv, (X_padded,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TfIdfVectorizer", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = cv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_unigrams_repeated_tokens(self):
        """Counts exceed 1 when a token appears multiple times."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        texts = ["apple apple banana", "banana banana cherry", "apple banana cherry cherry"]
        cv = CountVectorizer()
        cv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(cv, (X_padded,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = cv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_bigrams(self):
        """ngram_range=(1, 2) — unigrams and bigrams."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world", "world peace", "hello peace"]
        cv = CountVectorizer(ngram_range=(1, 2))
        cv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(cv, (X_padded,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = cv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_bigrams_only(self):
        """ngram_range=(2, 2) — bigrams only."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world peace", "world peace hello", "hello world hello"]
        cv = CountVectorizer(ngram_range=(2, 2))
        cv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(cv, (X_padded,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = cv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_binary_true(self):
        """binary=True clips all counts to 1."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        texts = ["apple apple banana", "banana banana cherry", "apple banana cherry cherry"]
        cv = CountVectorizer(binary=True)
        cv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(cv, (X_padded,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = cv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_larger_vocabulary(self):
        """Slightly larger vocabulary to exercise the mapping logic."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        texts = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat and the dog are friends",
        ]
        cv = CountVectorizer()
        cv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(cv, (X_padded,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = cv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_raises_for_char_analyzer(self):
        """analyzer='char' must raise NotImplementedError."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello", "world"]
        cv = CountVectorizer(analyzer="char")
        cv.fit(texts)

        X_padded = _pad_tokens(texts, tokenize=list)
        with self.assertRaises(NotImplementedError):
            to_onnx(cv, (X_padded,))

    def test_raises_for_non_string_input(self):
        """Float input must raise NotImplementedError."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world", "world peace"]
        cv = CountVectorizer()
        cv.fit(texts)

        X_float = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(NotImplementedError):
            to_onnx(cv, (X_float,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
