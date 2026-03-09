"""
Unit tests for yobx.sklearn.lightgbm LGBMRegressor converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_lightgbm
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
@requires_lightgbm("4.0")
class TestLGBMRegressor(ExtTestCase):
    def _make_regression_data(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = (X[:, 0] + 2 * X[:, 1]).astype(np.float32)
        return X, y

    def test_lgbm_regressor(self):
        from lightgbm import LGBMRegressor

        X, y = self._make_regression_data()
        reg = LGBMRegressor(n_estimators=5, max_depth=3, random_state=0, verbose=-1)
        reg.fit(X, y)

        onx = to_onnx(reg, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertTrue(
            any(t in op_types for t in ("TreeEnsembleRegressor", "TreeEnsemble")),
            f"Expected a tree node, got {op_types}",
        )

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = reg.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_lgbm_regressor_poisson_objective(self):
        """poisson objective applies exp — predictions are positive counts."""
        from lightgbm import LGBMRegressor

        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = rng.poisson(lam=3, size=50).astype(np.float32) + 1.0
        reg = LGBMRegressor(n_estimators=5, max_depth=3, random_state=0, verbose=-1,
                            objective="poisson")
        reg.fit(X, y)

        onx = to_onnx(reg, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Exp", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = reg.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_lgbm_regressor_tweedie_objective(self):
        """tweedie objective applies exp — predictions are positive."""
        from lightgbm import LGBMRegressor

        rng = np.random.default_rng(4)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = np.abs(rng.standard_normal(50)).astype(np.float32) + 0.5
        reg = LGBMRegressor(n_estimators=5, max_depth=3, random_state=0, verbose=-1,
                            objective="tweedie")
        reg.fit(X, y)

        onx = to_onnx(reg, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Exp", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = reg.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_lgbm_regressor_unsupported_objective_raises(self):
        """Unknown regression objectives raise NotImplementedError."""
        from yobx.sklearn.lightgbm.lgbm import _get_reg_output_transform

        with self.assertRaises(NotImplementedError):
            _get_reg_output_transform("unknown:objective")

    def test_lgbm_regressor_dtypes_opsets(self):
        from lightgbm import LGBMRegressor

        X32, y = self._make_regression_data()
        reg = LGBMRegressor(n_estimators=5, max_depth=3, random_state=0, verbose=-1)
        reg.fit(X32, y)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 21, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(reg, (X,), target_opset=target_opset)

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

                    expected = reg.predict(X32).astype(np.float32).astype(dtype).reshape(-1, 1)
                    self.assertEqualArray(expected, predictions, atol=1e-4)

    @requires_sklearn("1.8")
    def test_lgbm_regressor_pipeline(self):
        """LGBMRegressor works inside a sklearn Pipeline."""
        from lightgbm import LGBMRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = self._make_regression_data()
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", LGBMRegressor(n_estimators=3, random_state=0, verbose=-1)),
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
