"""
Parity tests for the TfidfVectorizer converter, mirroring sklearn-onnx's
``test_sklearn_tfidf_vectorizer_converter.py``.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


# Corpus from sklearn-onnx's test suite (natural-language sentences with
# punctuation so that sklearn's default word-boundary regex tokenizer is
# exercised via ``_sklearn_tokenize``).
_SKL2ONNX_CORPUS = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]


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


@requires_sklearn("1.4")
class TestSklearnTfidfVectorizerConverter(ExtTestCase):
    """Tests mirroring sklearn-onnx's TfidfVectorizer test suite.

    The corpus used here (``_SKL2ONNX_CORPUS``) is identical to the one in
    sklearn-onnx's ``test_sklearn_tfidf_vectorizer_converter.py``.  Documents
    contain punctuation, so sklearn's default word-boundary regex tokenizer
    is exercised via ``_sklearn_tokenize``.
    """

    def _tokenize(self, estimator):
        return _sklearn_tokenize(estimator, _SKL2ONNX_CORPUS)

    # ------------------------------------------------------------------
    # ngram_range tests (mirrors test_model_tfidf_vectorizer11/22/12/13)
    # ------------------------------------------------------------------

    def test_model_tfidf_vectorizer11(self):
        """Unigrams (ngram_range=(1,1)), norm=None on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        tv = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        tv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(tv)
        onx = to_onnx(tv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_model_tfidf_vectorizer22(self):
        """Bigrams only (ngram_range=(2,2)) on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        tv = TfidfVectorizer(ngram_range=(2, 2), norm=None)
        tv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(tv)
        onx = to_onnx(tv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_model_tfidf_vectorizer12_norm_none(self):
        """Unigrams and bigrams (ngram_range=(1,2)), norm=None on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        tv = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        tv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(tv)
        onx = to_onnx(tv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_model_tfidf_vectorizer12_normL1(self):
        """Unigrams and bigrams (ngram_range=(1,2)), norm='l1' on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        tv = TfidfVectorizer(ngram_range=(1, 2), norm="l1")
        tv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(tv)
        onx = to_onnx(tv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_model_tfidf_vectorizer12_normL2(self):
        """Unigrams and bigrams (ngram_range=(1,2)), norm='l2' on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        tv = TfidfVectorizer(ngram_range=(1, 2), norm="l2")
        tv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(tv)
        onx = to_onnx(tv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_model_tfidf_vectorizer13(self):
        """Unigrams to trigrams (ngram_range=(1,3)) on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        tv = TfidfVectorizer(ngram_range=(1, 3), norm=None)
        tv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(tv)
        onx = to_onnx(tv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    # ------------------------------------------------------------------
    # binary mode (mirrors test_model_tfidf_vectorizer_binary)
    # ------------------------------------------------------------------

    def test_model_tfidf_vectorizer_binary(self):
        """binary=True clips all counts to 1 on the sklearn-onnx corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        tv = TfidfVectorizer(binary=True)
        tv.fit(_SKL2ONNX_CORPUS)
        X = self._tokenize(tv)
        onx = to_onnx(tv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(_SKL2ONNX_CORPUS).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    # ------------------------------------------------------------------
    # Out-of-vocabulary handling (mirrors test_model_tfidf_vectorizer11_out_vocabulary)
    # ------------------------------------------------------------------

    def test_model_tfidf_vectorizer11_out_vocabulary(self):
        """OOV tokens are silently ignored; known tokens still score correctly."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        tv = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        tv.fit(_SKL2ONNX_CORPUS)

        oov_docs = [
            "AZZ ZZ This is the first document.",
            "BZZ ZZ This document is the second document.",
            "ZZZ ZZ And this is the third one.",
            "WZZ ZZ Is this the first document?",
        ]
        X = _sklearn_tokenize(tv, oov_docs)
        onx = to_onnx(tv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(oov_docs).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
