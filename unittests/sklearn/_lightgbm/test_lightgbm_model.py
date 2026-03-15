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
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = (X[:, 0] + 2 * X[:, 1]).astype(np.float32)
        return X, y

    def test_lgbm_model_float32(self):
        """LGBMModel with float32 input converts and predicts correctly."""
        from lightgbm import LGBMModel

        X, y = self._make_regression_data()
        model = LGBMModel(objective="regression", n_estimators=5, max_depth=3, random_state=0, verbose=-1)
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

    def test_lgbm_model_float64(self):
        """LGBMModel with float64 input converts and predicts correctly."""
        from lightgbm import LGBMModel

        X32, y = self._make_regression_data()
        X = X32.astype(np.float64)
        model = LGBMModel(objective="regression", n_estimators=5, max_depth=3, random_state=0, verbose=-1)
        model.fit(X32, y)

        onx = to_onnx(model, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = model.predict(X32).astype(np.float64).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_lgbm_model_dtypes_opsets(self):
        """LGBMModel: float32/float64 x ai.onnx.ml opset 3 and 5."""
        from lightgbm import LGBMModel

        X32, y = self._make_regression_data()
        model = LGBMModel(objective="regression", n_estimators=5, max_depth=3, random_state=0, verbose=-1)
        model.fit(X32, y)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 21, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(model, (X,), target_opset=target_opset)

                    ml_opsets = {op.domain: op.version for op in onx.opset_import}
                    self.assertEqual(ml_opsets.get("ai.onnx.ml"), ml_opset)

                    if ml_opset >= 5:
                        op_types = [n.op_type for n in onx.graph.node]
                        self.assertIn("TreeEnsemble", op_types)
                    else:
                        op_types = [n.op_type for n in onx.graph.node]
                        self.assertIn("TreeEnsembleRegressor", op_types)

                    ref = ExtendedReferenceEvaluator(onx)
                    results = ref.run(None, {"X": X})
                    predictions = results[0]

                    expected = model.predict(X32).astype(np.float32).astype(dtype).reshape(-1, 1)
                    self.assertEqualArray(expected, predictions, atol=1e-4)

    def test_lgbm_model_poisson_objective(self):
        """LGBMModel with poisson objective applies exp transform."""
        from lightgbm import LGBMModel

        rng = np.random.default_rng(3)
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

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_lgbm_model_unsupported_objective_raises(self):
        """Unknown objectives raise NotImplementedError."""
        from yobx.sklearn.lightgbm.lgbm import _get_reg_output_transform

        with self.assertRaises(NotImplementedError):
            _get_reg_output_transform("unknown:objective")

    @requires_sklearn("1.8")
    def test_lgbm_model_pipeline(self):
        """LGBMModel works inside a sklearn Pipeline."""
        from lightgbm import LGBMModel
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = self._make_regression_data()
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LGBMModel(objective="regression", n_estimators=3, random_state=0, verbose=-1)),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = pipe.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
