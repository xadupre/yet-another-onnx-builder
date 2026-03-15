"""
Unit tests for yobx.sklearn.lightgbm LGBMModel converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_lightgbm
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
@requires_lightgbm("4.0")
class TestLGBMModel(ExtTestCase):
    def _make_regression_data(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = (X[:, 0] + 2 * X[:, 1]).astype(np.float32)
        return X, y

    def _make_binary_data(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def _make_multiclass_data(self):
        rng = np.random.default_rng(12)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = np.array([i % 3 for i in range(60)])
        return X, y

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def test_lgbm_model_regression(self):
        """LGBMModel with regression objective converts correctly."""
        from lightgbm.sklearn import LGBMModel

        X, y = self._make_regression_data()
        model = LGBMModel(
            objective="regression", n_estimators=5, max_depth=3, random_state=0, verbose=-1
        )
        model.fit(X, y)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertTrue(
            any(t in op_types for t in ("TreeEnsembleRegressor", "TreeEnsemble")),
            f"Expected a tree node, got {op_types}",
        )

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = model.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_lgbm_model_regression_dtypes_opsets(self):
        """LGBMModel regression: float32/float64 x ai.onnx.ml opset 3 and 5."""
        from lightgbm.sklearn import LGBMModel

        X32, y = self._make_regression_data()
        model = LGBMModel(
            objective="regression", n_estimators=5, max_depth=3, random_state=0, verbose=-1
        )
        model.fit(X32, y)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 21, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(model, (X,), target_opset=target_opset)

                    ml_opsets = {op.domain: op.version for op in onx.opset_import}
                    self.assertEqual(ml_opsets.get("ai.onnx.ml"), ml_opset)

                    ref = ExtendedReferenceEvaluator(onx)
                    results = ref.run(None, {"X": X})
                    predictions = results[0]

                    expected = model.predict(X32).astype(np.float32).astype(dtype).reshape(-1, 1)
                    self.assertEqualArray(expected, predictions, atol=1e-4)

                    sess = self.check_ort(onx)
                    ort_results = sess.run(None, {"X": X})
                    self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_lgbm_model_poisson_objective(self):
        """LGBMModel with poisson objective applies exp transform."""
        from lightgbm.sklearn import LGBMModel

        rng = np.random.default_rng(13)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = rng.poisson(lam=3, size=50).astype(np.float32) + 1.0
        model = LGBMModel(
            objective="poisson", n_estimators=5, max_depth=3, random_state=0, verbose=-1
        )
        model.fit(X, y)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Exp", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = model.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

    def test_lgbm_model_unsupported_objective_raises(self):
        """Unknown objectives raise NotImplementedError."""
        from yobx.sklearn.lightgbm.lgbm import _get_reg_output_transform

        with self.assertRaises(NotImplementedError):
            _get_reg_output_transform("unknown:objective")

    # ------------------------------------------------------------------
    # Binary classification
    # ------------------------------------------------------------------

    def test_lgbm_model_binary(self):
        """LGBMModel with binary objective outputs sigmoid probabilities [N, 1]."""
        from lightgbm.sklearn import LGBMModel

        X, y = self._make_binary_data()
        model = LGBMModel(
            objective="binary", n_estimators=5, max_depth=3, random_state=0, verbose=-1
        )
        model.fit(X, y)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sigmoid", op_types)
        self.assertTrue(
            any(t in op_types for t in ("TreeEnsembleRegressor", "TreeEnsemble")),
            f"Expected a tree node, got {op_types}",
        )

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        # LGBMModel.predict() returns shape [N] probabilities; reshape to [N, 1]
        expected = model.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_lgbm_model_binary_dtypes_opsets(self):
        """LGBMModel binary: float32/float64 x ai.onnx.ml opset 3 and 5."""
        from lightgbm.sklearn import LGBMModel

        X32, y = self._make_binary_data()
        model = LGBMModel(
            objective="binary", n_estimators=5, max_depth=3, random_state=0, verbose=-1
        )
        model.fit(X32, y)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 21, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(model, (X,), target_opset=target_opset)

                    ml_opsets = {op.domain: op.version for op in onx.opset_import}
                    self.assertEqual(ml_opsets.get("ai.onnx.ml"), ml_opset)

                    ref = ExtendedReferenceEvaluator(onx)
                    results = ref.run(None, {"X": X})
                    predictions = results[0]

                    expected = model.predict(X32).astype(np.float32).astype(dtype).reshape(-1, 1)
                    self.assertEqualArray(expected, predictions, atol=1e-5)

                    sess = self.check_ort(onx)
                    ort_results = sess.run(None, {"X": X})
                    self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    # ------------------------------------------------------------------
    # Multi-class classification
    # ------------------------------------------------------------------

    def test_lgbm_model_multiclass(self):
        """LGBMModel with multiclass objective outputs softmax proba [N, n_classes]."""
        from lightgbm.sklearn import LGBMModel

        X, y = self._make_multiclass_data()
        model = LGBMModel(
            objective="multiclass",
            num_class=3,
            n_estimators=5,
            max_depth=3,
            random_state=0,
            verbose=-1,
        )
        model.fit(X, y)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Softmax", op_types)
        self.assertTrue(
            any(t in op_types for t in ("TreeEnsembleRegressor", "TreeEnsemble")),
            f"Expected a tree node, got {op_types}",
        )

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        # LGBMModel.predict() returns shape [N, n_classes] probabilities
        expected = model.predict(X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_lgbm_model_multiclass_dtypes_opsets(self):
        """LGBMModel multiclass: float32/float64 x ai.onnx.ml opset 3 and 5."""
        from lightgbm.sklearn import LGBMModel

        X32, y = self._make_multiclass_data()
        model = LGBMModel(
            objective="multiclass",
            num_class=3,
            n_estimators=5,
            max_depth=3,
            random_state=0,
            verbose=-1,
        )
        model.fit(X32, y)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 21, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(model, (X,), target_opset=target_opset)

                    ml_opsets = {op.domain: op.version for op in onx.opset_import}
                    self.assertEqual(ml_opsets.get("ai.onnx.ml"), ml_opset)

                    ref = ExtendedReferenceEvaluator(onx)
                    results = ref.run(None, {"X": X})
                    predictions = results[0]

                    # The converter always casts the tree output to itype (input dtype),
                    # so the output dtype matches the input dtype for all opsets.
                    expected = model.predict(X32).astype(np.float32).astype(dtype)
                    self.assertEqualArray(expected, predictions, atol=1e-5)

                    sess = self.check_ort(onx)
                    ort_results = sess.run(None, {"X": X})
                    self.assertEqualArray(expected, ort_results[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
