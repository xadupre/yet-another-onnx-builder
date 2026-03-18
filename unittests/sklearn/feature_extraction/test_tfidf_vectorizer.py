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
    return np.array(
        [t + [""] * (max_len - len(t)) for t in tokenized], dtype=object
    )


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
        self.assertIn("Mul", op_types)       # IDF weighting
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
