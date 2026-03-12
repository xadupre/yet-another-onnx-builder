"""
Unit tests for yobx.sklearn.preprocessing.KBinsDiscretizer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestKBinsDiscretizer(ExtTestCase):
    def _make_data(self):
        return np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)

    def test_ordinal_uniform(self):
        from sklearn.preprocessing import KBinsDiscretizer
        from yobx.sklearn import to_onnx

        X = self._make_data()
        kbd = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
        kbd.fit(X)

        onx = to_onnx(kbd, (X,))

        # Graph should use GreaterOrEqual and ReduceSum for bin indexing
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("GreaterOrEqual", op_types)
        self.assertIn("ReduceSum", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = kbd.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_ordinal_quantile(self):
        from sklearn.preprocessing import KBinsDiscretizer
        from yobx.sklearn import to_onnx

        X = self._make_data()
        kbd = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
        kbd.fit(X)

        onx = to_onnx(kbd, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = kbd.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_ordinal_different_bins_per_feature(self):
        from sklearn.preprocessing import KBinsDiscretizer
        from yobx.sklearn import to_onnx

        X = self._make_data()
        kbd = KBinsDiscretizer(n_bins=[2, 4], encode="ordinal", strategy="uniform")
        kbd.fit(X)

        onx = to_onnx(kbd, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = kbd.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_onehot_dense(self):
        from sklearn.preprocessing import KBinsDiscretizer
        from yobx.sklearn import to_onnx

        X = self._make_data()
        kbd = KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="uniform")
        kbd.fit(X)

        onx = to_onnx(kbd, (X,))

        # Graph should contain OneHot and Concat nodes
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("OneHot", op_types)
        self.assertIn("Concat", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = kbd.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_onehot_dense_different_bins(self):
        from sklearn.preprocessing import KBinsDiscretizer
        from yobx.sklearn import to_onnx

        X = self._make_data()
        kbd = KBinsDiscretizer(n_bins=[2, 4], encode="onehot-dense", strategy="uniform")
        kbd.fit(X)

        onx = to_onnx(kbd, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = kbd.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_onehot(self):
        from sklearn.preprocessing import KBinsDiscretizer
        from yobx.sklearn import to_onnx

        X = self._make_data()
        kbd = KBinsDiscretizer(n_bins=3, encode="onehot", strategy="uniform")
        kbd.fit(X)

        onx = to_onnx(kbd, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        # sklearn 'onehot' returns a sparse matrix; convert to dense for comparison
        expected = kbd.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_output_shape_ordinal(self):
        from sklearn.preprocessing import KBinsDiscretizer
        from yobx.sklearn import to_onnx

        X = self._make_data()
        kbd = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
        kbd.fit(X)
        onx = to_onnx(kbd, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqual(result.shape, (5, 2))

    def test_output_shape_onehot_dense(self):
        from sklearn.preprocessing import KBinsDiscretizer
        from yobx.sklearn import to_onnx

        X = self._make_data()
        kbd = KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="uniform")
        kbd.fit(X)
        onx = to_onnx(kbd, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        # 2 features × 3 bins = 6 one-hot columns
        self.assertEqual(result.shape, (5, 6))

    def test_single_feature(self):
        from sklearn.preprocessing import KBinsDiscretizer
        from yobx.sklearn import to_onnx

        X = np.array([[1], [3], [5], [7], [9]], dtype=np.float32)
        kbd = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
        kbd.fit(X)
        onx = to_onnx(kbd, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = kbd.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
