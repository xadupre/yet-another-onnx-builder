"""
Unit tests for yobx.sklearn.ensemble HistGradientBoosting converters.
"""

import unittest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnHistGradientBoosting(ExtTestCase):
    # ------------------------------------------------------------------ #
    # Regression                                                           #
    # ------------------------------------------------------------------ #

    def test_hgb_regressor_float32(self):
        """HGBRegressor, float32 input, legacy opset (ai.onnx.ml < 5)."""
        X = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float32,
        )
        y = np.array([1.0, 1.5, 2.0, 2.5, 8.0, 9.0, 9.5, 10.0], dtype=np.float32)
        est = HistGradientBoostingRegressor(max_iter=5, max_depth=3, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,))

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

    def test_hgb_regressor_float64(self):
        """HGBRegressor, float64 input, legacy opset (ai.onnx.ml < 5)."""
        X = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float64,
        )
        y = np.array([1.0, 1.5, 2.0, 2.5, 8.0, 9.0, 9.5, 10.0])
        est = HistGradientBoostingRegressor(max_iter=5, max_depth=3, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,))

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

    def test_hgb_regressor_float32_v5(self):
        """HGBRegressor, float32 input, ai.onnx.ml opset 5 (TreeEnsemble)."""
        X = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float32,
        )
        y = np.array([1.0, 1.5, 2.0, 2.5, 8.0, 9.0, 9.5, 10.0], dtype=np.float32)
        est = HistGradientBoostingRegressor(max_iter=5, max_depth=3, random_state=42)
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

    def test_hgb_regressor_float64_v5(self):
        """HGBRegressor, float64 input, ai.onnx.ml opset 5 (TreeEnsemble)."""
        X = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float64,
        )
        y = np.array([1.0, 1.5, 2.0, 2.5, 8.0, 9.0, 9.5, 10.0])
        est = HistGradientBoostingRegressor(max_iter=5, max_depth=3, random_state=42)
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

        expected = est.predict(X).reshape(-1, 1)
        self.assertEqualArray(expected, pred, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    # ------------------------------------------------------------------ #
    # Binary classification                                                #
    # ------------------------------------------------------------------ #

    def test_hgb_classifier_binary_float32(self):
        """HGBClassifier binary, float32 input, legacy opset."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float32,
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        est = HistGradientBoostingClassifier(max_iter=5, max_depth=3, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,))

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

    def test_hgb_classifier_binary_float64(self):
        """HGBClassifier binary, float64 input, legacy opset."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float64,
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        est = HistGradientBoostingClassifier(max_iter=5, max_depth=3, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,))

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)

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

    def test_hgb_classifier_binary_float32_v5(self):
        """HGBClassifier binary, float32 input, ai.onnx.ml opset 5."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float32,
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        est = HistGradientBoostingClassifier(max_iter=5, max_depth=3, random_state=42)
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

    def test_hgb_classifier_binary_float64_v5(self):
        """HGBClassifier binary, float64 input, ai.onnx.ml opset 5."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]],
            dtype=np.float64,
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        est = HistGradientBoostingClassifier(max_iter=5, max_depth=3, random_state=42)
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

    # ------------------------------------------------------------------ #
    # Multiclass classification                                            #
    # ------------------------------------------------------------------ #

    def test_hgb_classifier_multiclass_float32(self):
        """HGBClassifier multiclass, float32 input, legacy opset."""
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9],
             [1, 3], [3, 5]],
            dtype=np.float32,
        )
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        est = HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,))

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
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9],
             [1, 3], [3, 5]],
            dtype=np.float64,
        )
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        est = HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=42)
        est.fit(X, y)

        onx = to_onnx(est, (X,))

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)

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

    def test_hgb_classifier_multiclass_float32_v5(self):
        """HGBClassifier multiclass, float32 input, ai.onnx.ml opset 5."""
        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9],
             [1, 3], [3, 5]],
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
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [8, 9],
             [1, 3], [3, 5]],
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
        expected_proba = est.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    # ------------------------------------------------------------------ #
    # Numerical features (explicit)                                        #
    # ------------------------------------------------------------------ #

    def test_hgb_numerical_features_regressor(self):
        """HGBRegressor with explicitly numerical features only."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1] - X[:, 2]
        est = HistGradientBoostingRegressor(
            max_iter=5, max_depth=3, random_state=0
        )
        est.fit(X, y)

        onx = to_onnx(est, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        expected = est.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, results[0], atol=1e-5)

    def test_hgb_numerical_features_classifier(self):
        """HGBClassifier with explicitly numerical features only."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        est = HistGradientBoostingClassifier(
            max_iter=5, max_depth=3, random_state=1
        )
        est.fit(X, y)

        onx = to_onnx(est, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(est.predict(X), label)
        self.assertEqualArray(
            est.predict_proba(X).astype(np.float32), proba, atol=1e-5
        )

    # ------------------------------------------------------------------ #
    # Categorical features — should raise NotImplementedError             #
    # ------------------------------------------------------------------ #

    def test_hgb_categorical_features_raises(self):
        """Categorical splits are not supported and must raise NotImplementedError."""
        rng = np.random.RandomState(0)
        n = 200
        cat = rng.randint(0, 5, size=n).astype(np.float32)
        num = rng.randn(n).astype(np.float32)
        X = np.stack([cat, num], axis=1)
        y = (cat % 2 == 0).astype(int)
        est = HistGradientBoostingClassifier(
            max_iter=10,
            max_depth=3,
            categorical_features=[0],  # column 0 is categorical
            random_state=0,
        )
        est.fit(X, y)

        with self.assertRaises(NotImplementedError):
            to_onnx(est, (X,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
