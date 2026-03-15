"""
Unit tests for yobx.sklearn.lightgbm LGBMRanker converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_lightgbm
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
@requires_lightgbm("4.0")
class TestLGBMRanker(ExtTestCase):
    def _make_ranking_data(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        # Labels must be non-negative integers for ranking
        y = rng.integers(0, 5, size=50).astype(np.float32)
        group = [25, 25]
        return X, y, group

    def test_lgbm_ranker(self):
        from lightgbm import LGBMRanker

        X, y, group = self._make_ranking_data()
        ranker = LGBMRanker(n_estimators=5, max_depth=3, random_state=0, verbose=-1)
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

    def test_lgbm_ranker_rank_xendcg_objective(self):
        """rank_xendcg objective — identity link, no output transform."""
        from lightgbm import LGBMRanker

        X, y, group = self._make_ranking_data()
        ranker = LGBMRanker(
            n_estimators=5, max_depth=3, random_state=0, verbose=-1, objective="rank_xendcg"
        )
        ranker.fit(X, y, group=group)

        onx = to_onnx(ranker, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertNotIn("Exp", op_types)
        self.assertNotIn("Sigmoid", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = ranker.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_lgbm_ranker_unsupported_objective_raises(self):
        """Unknown ranking objectives raise NotImplementedError."""
        from unittest.mock import patch
        from lightgbm import LGBMRanker

        X, y, group = self._make_ranking_data()
        ranker = LGBMRanker(n_estimators=3, max_depth=2, random_state=0, verbose=-1)
        ranker.fit(X, y, group=group)

        # Patch the booster's dump_model to return an unsupported objective.
        original_dump = ranker.booster_.dump_model

        def patched_dump():
            result = original_dump()
            result["objective"] = "unknown_rank_obj"
            return result

        with (
            patch.object(ranker.booster_, "dump_model", side_effect=patched_dump),
            self.assertRaises(NotImplementedError),
        ):
            to_onnx(ranker, (X,))

    def test_lgbm_ranker_dtypes_opsets(self):
        """LGBMRanker: float32/float64 x ai.onnx.ml opset 3 and 5."""
        from lightgbm import LGBMRanker

        X32, y, group = self._make_ranking_data()
        ranker = LGBMRanker(n_estimators=5, max_depth=3, random_state=0, verbose=-1)
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

                    expected = ranker.predict(X32).astype(np.float32).astype(dtype).reshape(-1, 1)
                    self.assertEqualArray(expected, predictions, atol=1e-4)

                    sess = self.check_ort(onx)
                    ort_results = sess.run(None, {"X": X})
                    self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    @requires_sklearn("1.8")
    def test_lgbm_ranker_pipeline(self):
        """LGBMRanker works inside a sklearn Pipeline."""
        from lightgbm import LGBMRanker
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y, group = self._make_ranking_data()
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ranker", LGBMRanker(n_estimators=3, random_state=0, verbose=-1)),
            ]
        )
        pipe.fit(X, y, ranker__group=group)

        onx = to_onnx(pipe, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = pipe.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, predictions, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
