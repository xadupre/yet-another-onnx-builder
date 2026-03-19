import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestMissingIndicator(ExtTestCase):
    def test_missing_indicator_nan_float32(self):
        from sklearn.impute import MissingIndicator
        from yobx.sklearn import to_onnx

        X = np.array([[np.nan, 1], [2, np.nan], [3, 3]], dtype=np.float32)
        mi = MissingIndicator()
        mi.fit(X)

        onx = to_onnx(mi, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("IsNaN", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mi.transform(X)
        self.assertEqualArray(expected, result)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result)

    def test_missing_indicator_nan_float64(self):
        from sklearn.impute import MissingIndicator
        from yobx.sklearn import to_onnx

        X = np.array([[np.nan, 1], [2, np.nan], [3, 3]], dtype=np.float64)
        mi = MissingIndicator()
        mi.fit(X)

        onx = to_onnx(mi, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("IsNaN", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mi.transform(X)
        self.assertEqualArray(expected, result)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result)

    def test_missing_indicator_features_all_float32(self):
        from sklearn.impute import MissingIndicator
        from yobx.sklearn import to_onnx

        X = np.array([[np.nan, 1], [2, 2], [3, 3]], dtype=np.float32)
        mi = MissingIndicator(features="all")
        mi.fit(X)

        onx = to_onnx(mi, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mi.transform(X)
        self.assertEqualArray(expected, result)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result)

    def test_missing_indicator_features_all_float64(self):
        from sklearn.impute import MissingIndicator
        from yobx.sklearn import to_onnx

        X = np.array([[np.nan, 1], [2, 2], [3, 3]], dtype=np.float64)
        mi = MissingIndicator(features="all")
        mi.fit(X)

        onx = to_onnx(mi, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mi.transform(X)
        self.assertEqualArray(expected, result)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result)

    def test_missing_indicator_missing_only_subset_float32(self):
        """Only one of two features has missing values during training."""
        from sklearn.impute import MissingIndicator
        from yobx.sklearn import to_onnx

        X = np.array([[np.nan, 1], [2, 2], [3, 3]], dtype=np.float32)
        mi = MissingIndicator(features="missing-only")
        mi.fit(X)
        # features_ should only contain column 0
        self.assertEqual(list(mi.features_), [0])

        onx = to_onnx(mi, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Gather", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mi.transform(X)
        self.assertEqualArray(expected, result)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result)

    def test_missing_indicator_missing_only_subset_float64(self):
        """Only one of two features has missing values during training."""
        from sklearn.impute import MissingIndicator
        from yobx.sklearn import to_onnx

        X = np.array([[np.nan, 1], [2, 2], [3, 3]], dtype=np.float64)
        mi = MissingIndicator(features="missing-only")
        mi.fit(X)
        self.assertEqual(list(mi.features_), [0])

        onx = to_onnx(mi, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mi.transform(X)
        self.assertEqualArray(expected, result)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result)

    def test_missing_indicator_numeric_missing_value_float32(self):
        from sklearn.impute import MissingIndicator
        from yobx.sklearn import to_onnx

        X = np.array([[1, -1], [2, 3], [-1, 3]], dtype=np.float32)
        mi = MissingIndicator(missing_values=-1)
        mi.fit(X)

        onx = to_onnx(mi, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Equal", op_types)
        self.assertNotIn("IsNaN", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mi.transform(X)
        self.assertEqualArray(expected, result)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result)

    def test_missing_indicator_numeric_missing_value_float64(self):
        from sklearn.impute import MissingIndicator
        from yobx.sklearn import to_onnx

        X = np.array([[1, -1], [2, 3], [-1, 3]], dtype=np.float64)
        mi = MissingIndicator(missing_values=-1)
        mi.fit(X)

        onx = to_onnx(mi, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Equal", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mi.transform(X)
        self.assertEqualArray(expected, result)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
