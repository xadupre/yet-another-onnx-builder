"""
Unit tests for yobx.sklearn.xgboost XGBClassifier converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_xgboost
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
@requires_xgboost("3.0")
class TestXGBoostClassifier(ExtTestCase):
    def _make_binary_data(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def _make_multiclass_data(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = np.array([i % 3 for i in range(30)])
        return X, y

    def test_xgb_classifier_binary(self):
        from xgboost import XGBClassifier

        X, y = self._make_binary_data()
        clf = XGBClassifier(n_estimators=5, max_depth=3, random_state=0)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertTrue(
            any(t in op_types for t in ("TreeEnsembleClassifier", "TreeEnsemble")),
            f"Expected TreeEnsembleClassifier or TreeEnsemble, got {op_types}",
        )

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_proba = clf.predict_proba(X).astype(np.float32)
        expected_label = clf.predict(X)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)
        self.assertEqualArray(expected_label, label)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)
        self.assertEqualArray(expected_label, ort_results[0])

    def test_xgb_classifier_multiclass(self):
        from xgboost import XGBClassifier

        X, y = self._make_multiclass_data()
        clf = XGBClassifier(
            n_estimators=5,
            max_depth=3,
            random_state=0,
            eval_metric="mlogloss",
        )
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertTrue(
            any(t in op_types for t in ("TreeEnsembleClassifier", "TreeEnsemble")),
            f"Expected TreeEnsembleClassifier or TreeEnsemble, got {op_types}",
        )
        self.assertIn("Softmax", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_proba = clf.predict_proba(X).astype(np.float32)
        expected_label = clf.predict(X)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)
        self.assertEqualArray(expected_label, label)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)
        self.assertEqualArray(expected_label, ort_results[0])

    def test_xgb_classifier_binary_dtypes_opsets(self):
        """Binary classifier: float32/float64 × ai.onnx.ml opset 3 and 5."""
        from xgboost import XGBClassifier

        X32, y = self._make_binary_data()
        clf = XGBClassifier(n_estimators=5, max_depth=3, random_state=0)
        clf.fit(X32, y)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 20, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(clf, (X,), target_opset=target_opset)

                    ml_opsets = {op.domain: op.version for op in onx.opset_import}
                    self.assertEqual(ml_opsets.get("ai.onnx.ml"), ml_opset)

                    if ml_opset >= 5:
                        op_types = [n.op_type for n in onx.graph.node]
                        self.assertIn("TreeEnsemble", op_types)
                    else:
                        op_types = [n.op_type for n in onx.graph.node]
                        self.assertIn("TreeEnsembleClassifier", op_types)

                    ref = ExtendedReferenceEvaluator(onx)
                    results = ref.run(None, {"X": X})
                    proba, label = results[1], results[0]

                    # TreeEnsembleClassifier (opset ≤ 4) always outputs float32.
                    # TreeEnsemble (opset ≥ 5) outputs dtype when leaf weights are float64.
                    out_dtype = dtype if ml_opset >= 5 else np.float32
                    expected_proba = clf.predict_proba(X32).astype(out_dtype)
                    expected_label = clf.predict(X32)
                    self.assertEqualArray(expected_proba, proba, atol=1e-5)
                    self.assertEqualArray(expected_label, label)

    def test_xgb_classifier_multiclass_dtypes_opsets(self):
        """Multi-class classifier: float32/float64 × ai.onnx.ml opset 3 and 5."""
        from xgboost import XGBClassifier

        X32, y = self._make_multiclass_data()
        clf = XGBClassifier(
            n_estimators=5, max_depth=3, random_state=0, eval_metric="mlogloss"
        )
        clf.fit(X32, y)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 20, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(clf, (X,), target_opset=target_opset)

                    ml_opsets = {op.domain: op.version for op in onx.opset_import}
                    self.assertEqual(ml_opsets.get("ai.onnx.ml"), ml_opset)

                    ref = ExtendedReferenceEvaluator(onx)
                    results = ref.run(None, {"X": X})
                    proba, label = results[1], results[0]

                    out_dtype = dtype if ml_opset >= 5 else np.float32
                    expected_proba = clf.predict_proba(X32).astype(out_dtype)
                    expected_label = clf.predict(X32)
                    self.assertEqualArray(expected_proba, proba, atol=1e-5)
                    self.assertEqualArray(expected_label, label)

    def test_xgb_classifier_binary_pipeline(self):
        """XGBClassifier works inside a sklearn Pipeline."""
        from xgboost import XGBClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = self._make_binary_data()
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(n_estimators=3, random_state=0)),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_proba = pipe.predict_proba(X).astype(np.float32)
        expected_label = pipe.predict(X)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)
        self.assertEqualArray(expected_label, label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
