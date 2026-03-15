"""
Unit tests for yobx.sklearn.xgboost XGBRanker converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_xgboost
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
@requires_xgboost("3.0")
class TestXGBoostRanker(ExtTestCase):
    def _make_ranking_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = rng.integers(0, 3, size=30)
        # group: three queries of 10 samples each
        group = [10, 10, 10]
        return X, y, group

    def test_xgb_ranker_basic(self):
        """XGBRanker with default rank:pairwise objective."""
        from xgboost import XGBRanker

        X, y, group = self._make_ranking_data()
        ranker = XGBRanker(n_estimators=5, max_depth=3, random_state=0)
        ranker.fit(X, y, group=group)

        onx = to_onnx(ranker, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertTrue(
            any(t in op_types for t in ("TreeEnsembleRegressor", "TreeEnsemble")),
            f"Expected a tree node, got {op_types}",
        )

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = ranker.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_xgb_ranker_ndcg_objective(self):
        """XGBRanker with rank:ndcg objective."""
        from xgboost import XGBRanker

        X, y, group = self._make_ranking_data()
        ranker = XGBRanker(
            n_estimators=5, max_depth=3, random_state=0, objective="rank:ndcg"
        )
        ranker.fit(X, y, group=group)

        onx = to_onnx(ranker, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = ranker.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_xgb_ranker_map_objective(self):
        """XGBRanker with rank:map objective."""
        from xgboost import XGBRanker

        X, y, group = self._make_ranking_data()
        ranker = XGBRanker(
            n_estimators=5, max_depth=3, random_state=0, objective="rank:map"
        )
        ranker.fit(X, y, group=group)

        onx = to_onnx(ranker, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = ranker.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_xgb_ranker_unsupported_objective_raises(self):
        """Unknown ranking objectives raise NotImplementedError."""
        from yobx.sklearn.xgboost.xgb import _get_rank_output_transform

        with self.assertRaises(NotImplementedError):
            _get_rank_output_transform("unknown:objective")

    def test_xgb_ranker_dtypes_opsets(self):
        """XGBRanker: float32 and float64 inputs x ai.onnx.ml opset 3 and 5."""
        from xgboost import XGBRanker

        X32, y, group = self._make_ranking_data()
        ranker = XGBRanker(n_estimators=5, max_depth=3, random_state=0)
        ranker.fit(X32, y, group=group)

        for ml_opset in (3, 5):
            for dtype in (np.float32, np.float64):
                with self.subTest(ml_opset=ml_opset, dtype=dtype):
                    X = X32.astype(dtype)
                    target_opset = {"": 21, "ai.onnx.ml": ml_opset}
                    onx = to_onnx(ranker, (X,), target_opset=target_opset)

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

                    expected = (
                        ranker.predict(X32).astype(dtype).reshape(-1, 1)
                    )
                    self.assertEqualArray(expected, predictions, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
