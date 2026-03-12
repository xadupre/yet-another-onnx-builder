"""
Unit tests for yobx.sklearn.ensemble GradientBoosting converters.
"""

import unittest
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnGradientBoosting(ExtTestCase):
    # ------------------------------------------------------------------ #
    # Regression                                                           #
    # ------------------------------------------------------------------ #

    def test_gb_regressor_float32(self):
        """GBRegressor, float32 input, legacy opset (ai.onnx.ml < 5)."""
        X = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float32,
        )
        y = np.array([1.0, 1.5, 2.0, 2.5, 8.0, 9.0, 9.5, 10.0], dtype=np.float32)
        est = GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        pred = results[0]

        expected = est.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, pred, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_gb_regressor_float64(self):
        """GBRegressor, float64 input, legacy opset (ai.onnx.ml < 5)."""
        X = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float64,
        )
        y = np.array([1.0, 1.5, 2.0, 2.5, 8.0, 9.0, 9.5, 10.0])
        est = GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        pred = results[0]

        expected = est.predict(X).reshape(-1, 1)
        self.assertEqualArray(expected, pred, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_gb_regressor_float32_v5(self):
        """GBRegressor, float32 input, ai.onnx.ml opset 5 (TreeEnsemble)."""
        X = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float32,
        )
        y = np.array([1.0, 1.5, 2.0, 2.5, 8.0, 9.0, 9.5, 10.0], dtype=np.float32)
        est = GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        pred = results[0]

        expected = est.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, pred, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_gb_regressor_init_zero(self):
        """GBRegressor with init='zero', float32 input."""
        X = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float32,
        )
        y = np.array([1.0, 1.5, 2.0, 2.5, 8.0, 9.0, 9.5, 10.0], dtype=np.float32)
        est = GradientBoostingRegressor(n_estimators=5, max_depth=3, init="zero", random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        pred = results[0]

        expected = est.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, pred, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    # ------------------------------------------------------------------ #
    # Binary classification                                                #
    # ------------------------------------------------------------------ #

    def test_gb_classifier_binary_float32(self):
        """GBClassifier binary, float32 input, legacy opset."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float32,
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        est = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
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

    def test_gb_classifier_binary_float64(self):
        """GBClassifier binary, float64 input, legacy opset."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float64,
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        est = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

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

    def test_gb_classifier_binary_v5(self):
        """GBClassifier binary, float32 input, ai.onnx.ml opset 5."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float32,
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        est = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleRegressor", op_types)

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

    # ------------------------------------------------------------------ #
    # Multiclass classification                                            #
    # ------------------------------------------------------------------ #

    def test_gb_classifier_multiclass_float32(self):
        """GBClassifier multiclass (3 classes), float32 input, legacy opset."""
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5],
             [1, 3], [3, 5], [5, 7], [7, 9], [2, 4], [4, 6]],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
        est = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
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

    def test_gb_classifier_multiclass_v5(self):
        """GBClassifier multiclass, float32 input, ai.onnx.ml opset 5."""
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5],
             [1, 3], [3, 5], [5, 7], [7, 9], [2, 4], [4, 6]],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
        est = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleRegressor", op_types)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
