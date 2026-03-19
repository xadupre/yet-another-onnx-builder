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
    return np.array([t + [""] * (max_len - len(t)) for t in tokenized], dtype=object)


def _sklearn_tokenize(estimator, docs):
    """Pre-tokenize documents using the estimator's own sklearn tokenizer.

    Applies the same lowercasing, punctuation removal, and word-boundary
    splitting that sklearn uses internally so that the resulting tokens match
    the fitted vocabulary exactly.  Returns a rectangular 2-D object array
    padded with empty strings (as required by the yobx converters).
    """
    tokenize_fn = estimator.build_tokenizer()
    preprocessor = estimator.build_preprocessor()
    tokenized = [tokenize_fn(preprocessor(doc)) for doc in docs]
    max_len = max((len(t) for t in tokenized), default=1)
    max_len = max(max_len, 1)
    return np.array([t + [""] * (max_len - len(t)) for t in tokenized], dtype=object)


# Corpus from sklearn-onnx's test suite (natural-language sentences with
# punctuation so that sklearn's regex tokenizer is exercised).
_SKL2ONNX_CORPUS = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]


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


@requires_sklearn("1.4")
class TestCountVectorizerSkl2OnnxScenarios(ExtTestCase):
    """Tests mirroring sklearn-onnx's CountVectorizer test suite.

    The corpus used here (`_SKL2ONNX_CORPUS`) is identical to the one in
    sklearn-onnx's ``test_sklearn_count_vectorizer_converter.py``.  Documents
    contain punctuation, so sklearn's default word-boundary regex tokenizer
    is exercised via `_sklearn_tokenize`.
    """

    def _tokenize(self, estimator):
        return _sklearn_tokenize(estimator, _SKL2ONNX_CORPUS)

    # ------------------------------------------------------------------
    # ngram_range tests (mirrors test_model_count_vectorizer11/22/12/13)
    # ------------------------------------------------------------------

    def test_skl2onnx_corpus_ngram11(self):
        """Unigrams (ngram_range=(1,1)) on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        cv = CountVectorizer(ngram_range=(1, 1))
        cv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(cv)
        onx = to_onnx(cv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_skl2onnx_corpus_ngram22(self):
        """Bigrams only (ngram_range=(2,2)) on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        cv = CountVectorizer(ngram_range=(2, 2))
        cv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(cv)
        onx = to_onnx(cv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_skl2onnx_corpus_ngram12(self):
        """Unigrams and bigrams (ngram_range=(1,2)) on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        cv = CountVectorizer(ngram_range=(1, 2))
        cv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(cv)
        onx = to_onnx(cv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_skl2onnx_corpus_ngram13(self):
        """Unigrams to trigrams (ngram_range=(1,3)) on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        cv = CountVectorizer(ngram_range=(1, 3))
        cv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(cv)
        onx = to_onnx(cv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    # ------------------------------------------------------------------
    # binary mode (mirrors test_model_count_vectorizer_binary)
    # ------------------------------------------------------------------

    def test_skl2onnx_corpus_binary(self):
        """binary=True clips all counts to 1 on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        cv = CountVectorizer(binary=True)
        cv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(cv)
        onx = to_onnx(cv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    # ------------------------------------------------------------------
    # Out-of-vocabulary handling (mirrors test_model_tfidf_vectorizer11_out_vocabulary)
    # ------------------------------------------------------------------

    def test_skl2onnx_corpus_out_of_vocabulary(self):
        """OOV tokens are silently ignored; known tokens still count correctly."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        cv = CountVectorizer(ngram_range=(1, 1))
        cv.fit(_SKL2ONNX_CORPUS)

        oov_docs = [
            "AZZ ZZ This is the first document.",
            "BZZ ZZ This document is the second document.",
            "ZZZ ZZ And this is the third one.",
            "WZZ ZZ Is this the first document?",
        ]
        X = _sklearn_tokenize(cv, oov_docs)
        onx = to_onnx(cv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(oov_docs).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    # ------------------------------------------------------------------
    # Empty / sparse documents (mirrors test_model_tfidf_vectorizer11_empty_string_case1/2)
    # ------------------------------------------------------------------

    def test_skl2onnx_corpus_empty_string_case1(self):
        """A document consisting of only spaces produces an all-zero feature vector."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        cv = CountVectorizer(ngram_range=(1, 1))
        cv.fit(_SKL2ONNX_CORPUS[:3])

        test_docs = [_SKL2ONNX_CORPUS[2], " "]
        X = _sklearn_tokenize(cv, test_docs)
        onx = to_onnx(cv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(test_docs).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_skl2onnx_corpus_empty_string_case2(self):
        """An empty-string document produces an all-zero feature vector."""
        from sklearn.feature_extraction.text import CountVectorizer
        from yobx.sklearn import to_onnx

        cv = CountVectorizer(ngram_range=(1, 1))
        cv.fit(_SKL2ONNX_CORPUS)

        test_docs = list(_SKL2ONNX_CORPUS) + [""]
        X = _sklearn_tokenize(cv, test_docs)
        onx = to_onnx(cv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(test_docs).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
