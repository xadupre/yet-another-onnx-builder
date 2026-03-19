"""
Unit tests for yobx.sklearn.feature_extraction.TfidfVectorizer converter.
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
# punctuation so that sklearn's default word-boundary regex tokenizer
# is exercised via `_sklearn_tokenize`).
_SKL2ONNX_CORPUS = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]


@requires_sklearn("1.4")
class TestTfidfVectorizer(ExtTestCase):
    def test_default_float32(self):
        """Default TfidfVectorizer: use_idf=True, norm='l2', sublinear_tf=False."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world", "world peace", "hello peace"]
        tv = TfidfVectorizer()
        tv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(tv, (X_padded,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TfIdfVectorizer", op_types)
        self.assertIn("Mul", op_types)  # IDF weighting
        self.assertIn("ReduceL2", op_types)  # L2 normalisation

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_no_idf(self):
        """use_idf=False: only term-frequency + L2 normalisation."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world", "world peace", "hello peace"]
        tv = TfidfVectorizer(use_idf=False, norm="l2")
        tv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(tv, (X_padded,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_sublinear_tf(self):
        """sublinear_tf=True: replace counts with 1 + log(count)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        texts = ["apple apple banana", "banana banana cherry", "apple banana cherry cherry"]
        tv = TfidfVectorizer(sublinear_tf=True, use_idf=True, norm="l2")
        tv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(tv, (X_padded,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Log", op_types)
        self.assertIn("Where", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_norm_l1(self):
        """norm='l1' normalisation."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world", "world peace", "hello peace"]
        tv = TfidfVectorizer(norm="l1")
        tv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(tv, (X_padded,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("ReduceL1", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_no_norm(self):
        """norm=None: no row normalisation."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world", "world peace", "hello peace"]
        tv = TfidfVectorizer(norm=None)
        tv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(tv, (X_padded,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_bigrams(self):
        """ngram_range=(1, 2) — unigrams and bigrams."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world peace", "world peace hello", "hello world hello"]
        tv = TfidfVectorizer(ngram_range=(1, 2))
        tv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(tv, (X_padded,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_smooth_idf_false(self):
        """smooth_idf=False: IDF without +1 smoothing."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world", "world peace", "hello peace"]
        tv = TfidfVectorizer(smooth_idf=False)
        tv.fit(texts)

        X_padded = _pad_tokens(texts)
        onx = to_onnx(tv, (X_padded,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_padded})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_padded})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_raises_for_char_analyzer(self):
        """analyzer='char' must raise NotImplementedError."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello", "world"]
        tv = TfidfVectorizer(analyzer="char")
        tv.fit(texts)

        X_padded = np.array([list("hello"), list("world")], dtype=object)
        with self.assertRaises(NotImplementedError):
            to_onnx(tv, (X_padded,))

    def test_raises_for_non_string_input(self):
        """Float input must raise NotImplementedError."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world", "world peace"]
        tv = TfidfVectorizer()
        tv.fit(texts)

        X_float = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(NotImplementedError):
            to_onnx(tv, (X_float,))

    def test_low_opset_raises(self):
        """norm='l2' or 'l1' with opset < 18 must raise NotImplementedError."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yobx.sklearn import to_onnx

        texts = ["hello world", "world peace", "hello peace"]
        for norm in ("l1", "l2"):
            tv = TfidfVectorizer(norm=norm)
            tv.fit(texts)
            X_padded = _pad_tokens(texts)
            with self.assertRaises(NotImplementedError):
                to_onnx(tv, (X_padded,), target_opset={"": 17})


@requires_sklearn("1.4")
class TestTfidfVectorizerSkl2OnnxScenarios(ExtTestCase):
    """Tests mirroring sklearn-onnx's TfidfVectorizer test suite.

    The corpus used here (`_SKL2ONNX_CORPUS`) is identical to the one in
    sklearn-onnx's ``test_sklearn_tfidf_vectorizer_converter.py``.  Documents
    contain punctuation, so sklearn's default word-boundary regex tokenizer
    is exercised via `_sklearn_tokenize`.
    """

    def _tokenize(self, estimator):
        return _sklearn_tokenize(estimator, _SKL2ONNX_CORPUS)

    # ------------------------------------------------------------------
    # ngram_range tests (mirrors test_model_tfidf_vectorizer11/22/12/13)
    # ------------------------------------------------------------------

    def test_skl2onnx_corpus_ngram11_norm_none(self):
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

    def test_skl2onnx_corpus_ngram22(self):
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

    def test_skl2onnx_corpus_ngram12_norm_none(self):
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

    def test_skl2onnx_corpus_ngram12_norm_l1(self):
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

    def test_skl2onnx_corpus_ngram12_norm_l2(self):
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

    def test_skl2onnx_corpus_ngram13(self):
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

    def test_skl2onnx_corpus_binary(self):
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

    def test_skl2onnx_corpus_out_of_vocabulary(self):
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
