"""
Unit tests for yobx.sklearn.lightgbm LGBMClassifier converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_lightgbm
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
@requires_lightgbm("4.0")
class TestLGBMClassifier(ExtTestCase):
    def _make_binary_data(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def _make_multiclass_data(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = np.array([i % 3 for i in range(60)])
        return X, y

    def test_lgbm_classifier_binary(self):
        from lightgbm import LGBMClassifier

        X, y = self._make_binary_data()
        clf = LGBMClassifier(n_estimators=5, max_depth=3, random_state=0, verbose=-1)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
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

    def test_lgbm_classifier_multiclass(self):
        from lightgbm import LGBMClassifier

        X, y = self._make_multiclass_data()
        clf = LGBMClassifier(n_estimators=5, max_depth=3, random_state=0, verbose=-1)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
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

    def test_lgbm_classifier_binary_dtypes_opsets(self):
        from lightgbm import LGBMClassifier

        X32, y = self._make_binary_data()
        clf = LGBMClassifier(n_estimators=5, max_depth=3, random_state=0, verbose=-1)
        clf.fit(X32, y)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 20, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(clf, (X,), target_opset=target_opset)

                    ml_opsets = {op.domain: op.version for op in onx.proto.opset_import}
                    self.assertEqual(ml_opsets.get("ai.onnx.ml"), ml_opset)

                    if ml_opset >= 5:
                        op_types = [n.op_type for n in onx.proto.graph.node]
                        self.assertIn("TreeEnsemble", op_types)
                    else:
                        op_types = [n.op_type for n in onx.proto.graph.node]
                        self.assertIn("TreeEnsembleClassifier", op_types)

                    ref = ExtendedReferenceEvaluator(onx)
                    results = ref.run(None, {"X": X})
                    proba, label = results[1], results[0]

                    # Opset >= 5 (TreeEnsemble) natively supports float64 weights
                    # so probabilities match the input dtype; opset < 5 always
                    # uses float32 weights in TreeEnsembleClassifier.
                    out_dtype = dtype if ml_opset >= 5 else np.float32
                    expected_proba = clf.predict_proba(X32).astype(out_dtype)
                    expected_label = clf.predict(X32)
                    self.assertEqualArray(expected_proba, proba, atol=1e-5)
                    self.assertEqualArray(expected_label, label)

    def test_lgbm_classifier_multiclass_dtypes_opsets(self):
        """Multi-class classifier: float32/float64 x ai.onnx.ml opset 3 and 5."""
        from lightgbm import LGBMClassifier

        X32, y = self._make_multiclass_data()
        clf = LGBMClassifier(n_estimators=5, max_depth=3, random_state=0, verbose=-1)
        clf.fit(X32, y)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 20, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(clf, (X,), target_opset=target_opset)

                    ml_opsets = {op.domain: op.version for op in onx.proto.opset_import}
                    self.assertEqual(ml_opsets.get("ai.onnx.ml"), ml_opset)

                    ref = ExtendedReferenceEvaluator(onx)
                    results = ref.run(None, {"X": X})
                    proba, label = results[1], results[0]

                    # Opset >= 5 (TreeEnsemble) natively supports float64 weights
                    # so probabilities match the input dtype; opset < 5 always
                    # uses float32 weights in TreeEnsembleClassifier.
                    out_dtype = dtype if ml_opset >= 5 else np.float32
                    expected_proba = clf.predict_proba(X32).astype(out_dtype)
                    expected_label = clf.predict(X32)
                    self.assertEqualArray(expected_proba, proba, atol=1e-5)
                    self.assertEqualArray(expected_label, label)

    @requires_sklearn("1.8")
    def test_lgbm_classifier_binary_pipeline(self):
        from lightgbm import LGBMClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = self._make_binary_data()
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LGBMClassifier(n_estimators=3, random_state=0, verbose=-1)),
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

    def _make_categorical_data(self, n_classes=2):
        rng = np.random.default_rng(9)
        n = 300
        X_num = rng.standard_normal((n, 3)).astype(np.float32)
        cat = rng.integers(0, 5, size=n).astype(np.float32)
        X = np.column_stack([X_num, cat]).astype(np.float32)
        if n_classes == 2:
            y = (X_num[:, 0] + cat * 0.3 > 0).astype(int)
        else:
            y = np.array([i % n_classes for i in range(n)])
        return X, y

    def test_lgbm_classifier_binary_categorical(self):
        """Binary classifier with categorical features converts correctly."""
        from lightgbm import LGBMClassifier

        X, y = self._make_categorical_data(n_classes=2)
        clf = LGBMClassifier(
            n_estimators=10, max_depth=4, random_state=0, verbose=-1, min_child_samples=5
        )
        clf.fit(X, y, categorical_feature=[3])

        onx = to_onnx(clf, (X,))
        sess = self.check_ort(onx)
        results = sess.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_proba = clf.predict_proba(X).astype(np.float32)
        expected_label = clf.predict(X)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)
        self.assertEqualArray(expected_label, label)

    def test_lgbm_classifier_multiclass_categorical(self):
        """Multi-class classifier with categorical features converts correctly."""
        from lightgbm import LGBMClassifier

        X, y = self._make_categorical_data(n_classes=3)
        clf = LGBMClassifier(
            n_estimators=10, max_depth=4, random_state=0, verbose=-1, min_child_samples=5
        )
        clf.fit(X, y, categorical_feature=[3])

        onx = to_onnx(clf, (X,))
        sess = self.check_ort(onx)
        results = sess.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_proba = clf.predict_proba(X).astype(np.float32)
        expected_label = clf.predict(X)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)
        self.assertEqualArray(expected_label, label)

    def test_lgbm_classifier_categorical_dtypes_opsets(self):
        """Categorical classifier: float32/float64 x ai.onnx.ml opset 3 and 5."""
        from lightgbm import LGBMClassifier

        X32, y = self._make_categorical_data(n_classes=2)
        clf = LGBMClassifier(
            n_estimators=10, max_depth=4, random_state=0, verbose=-1, min_child_samples=5
        )
        clf.fit(X32, y, categorical_feature=[3])

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 21, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(clf, (X,), target_opset=target_opset)

                    ml_opsets = {op.domain: op.version for op in onx.proto.opset_import}
                    self.assertEqual(ml_opsets.get("ai.onnx.ml"), ml_opset)

                    sess = self.check_ort(onx)
                    ort_results = sess.run(None, {"X": X})
                    label, proba = ort_results[0], ort_results[1]

                    # Opset >= 5 (TreeEnsemble) natively supports float64 weights
                    # so probabilities match the input dtype; opset < 5 always
                    # uses float32 weights in TreeEnsembleClassifier.
                    out_dtype = dtype if ml_opset >= 5 else np.float32
                    expected_proba = clf.predict_proba(X32).astype(out_dtype)
                    expected_label = clf.predict(X32)
                    self.assertEqualArray(expected_proba, proba, atol=1e-5)
                    self.assertEqualArray(expected_label, label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
