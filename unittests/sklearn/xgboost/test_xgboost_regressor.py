"""
Unit tests for yobx.sklearn.xgboost XGBRegressor converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_xgboost
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
@requires_xgboost("3.0")
class TestXGBoostRegressor(ExtTestCase):
    def _make_regression_data(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = (X[:, 0] + 2 * X[:, 1]).astype(np.float32)
        return X, y

    def test_xgb_regressor(self):
        from xgboost import XGBRegressor

        X, y = self._make_regression_data()
        reg = XGBRegressor(n_estimators=5, max_depth=3, random_state=0)
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

    def test_xgb_regressor_logistic_objective(self):
        """reg:logistic objective applies sigmoid — predictions in (0, 1)."""
        from xgboost import XGBRegressor

        X, _ = self._make_regression_data()
        y = (X[:, 0] > 0).astype(np.float32)
        reg = XGBRegressor(
            n_estimators=5, max_depth=3, random_state=0, objective="reg:logistic"
        )
        reg.fit(X, y)

        onx = to_onnx(reg, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sigmoid", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = reg.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_xgb_regressor_poisson_objective(self):
        """count:poisson objective applies exp — predictions are positive counts."""
        from xgboost import XGBRegressor

        rng = np.random.default_rng(3)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = rng.poisson(lam=3, size=20).astype(np.float32)
        reg = XGBRegressor(
            n_estimators=5, max_depth=3, random_state=0, objective="count:poisson"
        )
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

    def test_xgb_regressor_unsupported_objective_raises(self):
        """Unknown regression objectives raise NotImplementedError."""
        from xgboost import XGBRegressor
        from yobx.sklearn.xgboost.xgb import _get_reg_output_transform

        with self.assertRaises(NotImplementedError):
            _get_reg_output_transform("unknown:objective")

    def test_xgb_regressor_dtypes_opsets(self):
        """Regressor: float32/float64 × ai.onnx.ml opset 3 and 5."""
        from xgboost import XGBRegressor

        X32, y = self._make_regression_data()
        reg = XGBRegressor(n_estimators=5, max_depth=3, random_state=0)
        reg.fit(X32, y)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 20, "ai.onnx.ml": ml_opset}
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

                    # When input is float64 the tree node outputs float64, so the
                    # ONNX model output is float64.  Cast the expected values to match.
                    expected = reg.predict(X32).astype(np.float32).astype(dtype).reshape(-1, 1)
                    self.assertEqualArray(expected, predictions, atol=1e-4)

    def test_xgb_regressor_pipeline(self):
        """XGBRegressor works inside a sklearn Pipeline."""
        from xgboost import XGBRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = self._make_regression_data()
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", XGBRegressor(n_estimators=3, random_state=0)),
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
