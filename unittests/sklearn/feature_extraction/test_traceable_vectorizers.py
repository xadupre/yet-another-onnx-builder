"""
Unit tests for TraceableCountVectorizer and TraceableTfIdfVectorizer.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestTraceableCountVectorizer(ExtTestCase):
    def test_unigrams_lowercase(self):
        """Raw text input; lowercase=True; unigrams."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableCountVectorizer

        texts = ["Hello World", "World PEACE", "Hello Peace"]
        cv = TraceableCountVectorizer(lowercase=True)
        cv.fit(texts)

        X = np.array(texts, dtype=object)
        onx = to_onnx(cv, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("StringNormalizer", op_types)
        self.assertIn("StringSplit", op_types)
        self.assertIn("TfIdfVectorizer", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_no_lowercase(self):
        """lowercase=False: StringNormalizer should not be emitted."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableCountVectorizer

        texts = ["hello world", "world peace", "hello peace"]
        cv = TraceableCountVectorizer(lowercase=False)
        cv.fit(texts)

        X = np.array(texts, dtype=object)
        onx = to_onnx(cv, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertNotIn("StringNormalizer", op_types)
        self.assertIn("StringSplit", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_bigrams(self):
        """ngram_range=(1, 2); raw text input."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableCountVectorizer

        texts = ["hello world", "world peace", "hello peace"]
        cv = TraceableCountVectorizer(ngram_range=(1, 2))
        cv.fit(texts)

        X = np.array(texts, dtype=object)
        onx = to_onnx(cv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_binary_true(self):
        """binary=True: duplicate tokens clipped to 1."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableCountVectorizer

        texts = ["apple apple banana", "banana banana cherry", "apple banana cherry cherry"]
        cv = TraceableCountVectorizer(binary=True)
        cv.fit(texts)

        X = np.array(texts, dtype=object)
        onx = to_onnx(cv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = cv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_raises_for_char_analyzer(self):
        """analyzer='char' must raise NotImplementedError."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableCountVectorizer

        texts = ["hello", "world"]
        cv = TraceableCountVectorizer(analyzer="char")
        cv.fit(texts)

        X = np.array(texts, dtype=object)
        with self.assertRaises(NotImplementedError):
            to_onnx(cv, (X,))

    def test_raises_for_non_string_input(self):
        """Float input must raise NotImplementedError."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableCountVectorizer

        texts = ["hello world", "world peace"]
        cv = TraceableCountVectorizer()
        cv.fit(texts)

        X_float = np.zeros((2,), dtype=np.float32)
        with self.assertRaises(NotImplementedError):
            to_onnx(cv, (X_float,))

    def test_raises_for_low_opset(self):
        """opset < 20 must raise NotImplementedError (StringSplit)."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableCountVectorizer

        texts = ["hello world", "world peace"]
        cv = TraceableCountVectorizer()
        cv.fit(texts)

        X = np.array(texts, dtype=object)
        with self.assertRaises(NotImplementedError):
            to_onnx(cv, (X,), target_opset={"": 19})


@requires_sklearn("1.4")
class TestTraceableTfIdfVectorizer(ExtTestCase):
    def test_default(self):
        """Default TF-IDF settings: use_idf=True, norm='l2', sublinear_tf=False."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableTfIdfVectorizer

        texts = ["hello world", "world peace", "hello peace"]
        tv = TraceableTfIdfVectorizer()
        tv.fit(texts)

        X = np.array(texts, dtype=object)
        onx = to_onnx(tv, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("StringSplit", op_types)
        self.assertIn("TfIdfVectorizer", op_types)
        self.assertIn("Mul", op_types)       # IDF weighting
        self.assertIn("ReduceL2", op_types)  # L2 normalisation

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_sublinear_tf(self):
        """sublinear_tf=True: replace counts with 1 + log(count)."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableTfIdfVectorizer

        texts = ["apple apple banana", "banana banana cherry", "apple banana cherry cherry"]
        tv = TraceableTfIdfVectorizer(sublinear_tf=True, use_idf=True, norm="l2")
        tv.fit(texts)

        X = np.array(texts, dtype=object)
        onx = to_onnx(tv, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Log", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_norm_l1(self):
        """norm='l1' normalisation."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableTfIdfVectorizer

        texts = ["hello world", "world peace", "hello peace"]
        tv = TraceableTfIdfVectorizer(norm="l1")
        tv.fit(texts)

        X = np.array(texts, dtype=object)
        onx = to_onnx(tv, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("ReduceL1", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_no_idf_no_norm(self):
        """use_idf=False, norm=None."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableTfIdfVectorizer

        texts = ["hello world", "world peace", "hello peace"]
        tv = TraceableTfIdfVectorizer(use_idf=False, norm=None)
        tv.fit(texts)

        X = np.array(texts, dtype=object)
        onx = to_onnx(tv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_bigrams(self):
        """ngram_range=(1, 2)."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableTfIdfVectorizer

        texts = ["hello world peace", "world peace hello", "hello world hello"]
        tv = TraceableTfIdfVectorizer(ngram_range=(1, 2))
        tv.fit(texts)

        X = np.array(texts, dtype=object)
        onx = to_onnx(tv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tv.transform(texts).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_raises_for_low_opset(self):
        """opset < 20 must raise NotImplementedError."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableTfIdfVectorizer

        texts = ["hello world", "world peace"]
        tv = TraceableTfIdfVectorizer()
        tv.fit(texts)

        X = np.array(texts, dtype=object)
        with self.assertRaises(NotImplementedError):
            to_onnx(tv, (X,), target_opset={"": 19})


if __name__ == "__main__":
    unittest.main(verbosity=2)
