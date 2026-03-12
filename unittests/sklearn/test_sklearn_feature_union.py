"""
Unit tests for yobx.sklearn FeatureUnion converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnFeatureUnion(ExtTestCase):
    def test_feature_union_two_scalers(self):
        X = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=np.float32,
        )
        fu = FeatureUnion([("std", StandardScaler()), ("mm", MinMaxScaler())])
        fu.fit(X)

        onx = to_onnx(fu, (X,))

        # Both transformers run on full X and are concatenated
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Concat", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = fu.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)
        # output should have 6 columns (3 from each transformer)
        self.assertEqual(result.shape[1], 6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_feature_union_single_transformer(self):
        X = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=np.float32,
        )
        fu = FeatureUnion([("std", StandardScaler())])
        fu.fit(X)

        onx = to_onnx(fu, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = fu.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_feature_union_drop_transformer(self):
        X = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=np.float32,
        )
        fu = FeatureUnion([("std", StandardScaler()), ("drop", "drop")])
        fu.fit(X)

        onx = to_onnx(fu, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = fu.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)
        # only the StandardScaler output (3 cols)
        self.assertEqual(result.shape[1], 3)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_pipeline_with_feature_union(self):
        X = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [2, 3, 4],
                [5, 6, 7],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1, 0, 1])

        fu = FeatureUnion([("std", StandardScaler()), ("mm", MinMaxScaler())])
        pipe = Pipeline([("features", fu), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_feature_union_nested(self):
        """FeatureUnion inside another FeatureUnion."""
        X = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=np.float32,
        )
        inner = FeatureUnion([("std", StandardScaler()), ("mm", MinMaxScaler())])
        outer = FeatureUnion([("inner", inner), ("std2", StandardScaler())])
        outer.fit(X)

        onx = to_onnx(outer, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = outer.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
