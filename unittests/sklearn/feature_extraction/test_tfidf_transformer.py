"""
Unit tests for yobx.sklearn.feature_extraction.TfidfTransformer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestTfidfTransformer(ExtTestCase):
    def _make_count_matrix(self, dtype=np.float32, seed=42):
        rng = np.random.default_rng(seed)
        # Non-negative integer-valued counts in float format
        return rng.integers(0, 5, size=(10, 8)).astype(dtype)

    def test_tfidf_default_float32(self):
        """Default settings: use_idf=True, norm='l2', sublinear_tf=False."""
        from sklearn.feature_extraction.text import TfidfTransformer
        from yobx.sklearn import to_onnx

        X = self._make_count_matrix(np.float32)
        tt = TfidfTransformer()
        tt.fit(X)

        onx = to_onnx(tt, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Mul", op_types)
        self.assertIn("ReduceL2", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tt.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_tfidf_default_float64(self):
        """Default settings with float64 input."""
        from sklearn.feature_extraction.text import TfidfTransformer
        from yobx.sklearn import to_onnx

        X = self._make_count_matrix(np.float64)
        tt = TfidfTransformer()
        tt.fit(X)

        onx = to_onnx(tt, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tt.transform(X).toarray().astype(np.float64)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-10)

    def test_tfidf_no_idf(self):
        """use_idf=False: only term-frequency + normalisation."""
        from sklearn.feature_extraction.text import TfidfTransformer
        from yobx.sklearn import to_onnx

        X = self._make_count_matrix(np.float32)
        tt = TfidfTransformer(use_idf=False, norm="l2")
        tt.fit(X)

        onx = to_onnx(tt, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tt.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_tfidf_sublinear_tf(self):
        """sublinear_tf=True: replace non-zero counts with 1+log(count)."""
        from sklearn.feature_extraction.text import TfidfTransformer
        from yobx.sklearn import to_onnx

        X = self._make_count_matrix(np.float32)
        # Ensure there are some non-zero entries
        X[0, 0] = 1.0
        tt = TfidfTransformer(use_idf=True, norm="l2", sublinear_tf=True)
        tt.fit(X)

        onx = to_onnx(tt, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Log", op_types)
        self.assertIn("Where", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tt.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_tfidf_norm_l1(self):
        """norm='l1' normalisation."""
        from sklearn.feature_extraction.text import TfidfTransformer
        from yobx.sklearn import to_onnx

        X = self._make_count_matrix(np.float32)
        tt = TfidfTransformer(use_idf=True, norm="l1", sublinear_tf=False)
        tt.fit(X)

        onx = to_onnx(tt, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("ReduceL1", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tt.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_tfidf_no_norm(self):
        """norm=None: no row normalisation."""
        from sklearn.feature_extraction.text import TfidfTransformer
        from yobx.sklearn import to_onnx

        X = self._make_count_matrix(np.float32)
        tt = TfidfTransformer(use_idf=True, norm=None, sublinear_tf=False)
        tt.fit(X)

        onx = to_onnx(tt, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tt.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_tfidf_sublinear_no_idf_no_norm(self):
        """sublinear_tf=True, use_idf=False, norm=None."""
        from sklearn.feature_extraction.text import TfidfTransformer
        from yobx.sklearn import to_onnx

        X = self._make_count_matrix(np.float32)
        tt = TfidfTransformer(use_idf=False, norm=None, sublinear_tf=True)
        tt.fit(X)

        onx = to_onnx(tt, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tt.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_tfidf_zero_row(self):
        """A row with all-zero counts should stay all-zero after normalisation."""
        from sklearn.feature_extraction.text import TfidfTransformer
        from yobx.sklearn import to_onnx

        X = np.array(
            [[1.0, 2.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32
        )
        tt = TfidfTransformer(use_idf=True, norm="l2", sublinear_tf=False)
        tt.fit(X)

        onx = to_onnx(tt, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tt.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_tfidf_smooth_idf_false(self):
        """smooth_idf=False: no +1 smoothing in idf computation."""
        from sklearn.feature_extraction.text import TfidfTransformer
        from yobx.sklearn import to_onnx

        X = self._make_count_matrix(np.float32)
        tt = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=False)
        tt.fit(X)

        onx = to_onnx(tt, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = tt.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_tfidf_low_opset_raises(self):
        """norm='l1' or 'l2' with opset < 18 must raise NotImplementedError."""
        from sklearn.feature_extraction.text import TfidfTransformer
        from yobx.sklearn import to_onnx

        X = self._make_count_matrix(np.float32)
        for norm in ("l1", "l2"):
            tt = TfidfTransformer(use_idf=True, norm=norm)
            tt.fit(X)
            with self.assertRaises(NotImplementedError):
                to_onnx(tt, (X,), target_opset={"": 17})


if __name__ == "__main__":
    unittest.main(verbosity=2)
