"""
Unit tests for yobx.sklearn.ensemble HistGradientBoosting converters.
"""

import unittest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnHistGradientBoostingMultiClass(ExtTestCase):
    def test_hgb_classifier_multiclass_float32(self):
        """HGBClassifier multiclass, float32 input, legacy opset."""
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9], [1, 3], [3, 5]],
            dtype=np.float32,
        )
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        est = HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = est.predict(X)
        expected_proba = est.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_hgb_classifier_multiclass_float64(self):
        """HGBClassifier multiclass, float64 input, legacy opset."""
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9], [1, 3], [3, 5]],
            dtype=np.float64,
        )
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        est = HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = est.predict(X)
        expected_proba = est.predict_proba(X)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_hgb_classifier_multiclass_float32_v5(self):
        """HGBClassifier multiclass, float32 input, ai.onnx.ml opset 5."""
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9], [1, 3], [3, 5]],
            dtype=np.float32,
        )
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        est = HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = est.predict(X)
        expected_proba = est.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_hgb_classifier_multiclass_float64_v5(self):
        """HGBClassifier multiclass, float64 input, ai.onnx.ml opset 5."""
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9], [1, 3], [3, 5]],
            dtype=np.float64,
        )
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        est = HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = est.predict(X)
        expected_proba = est.predict_proba(X)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
