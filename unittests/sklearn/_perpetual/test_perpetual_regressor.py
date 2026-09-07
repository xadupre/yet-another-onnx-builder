import unittest
import json
from unittest.mock import patch
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_perpetual, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from ._helpers import native_to_onnx


@requires_sklearn("1.4")
@requires_perpetual("2.1")
class TestPerpetualRegressor(ExtTestCase):
    def test_perpetual_regressor(self):
        from perpetual import PerpetualRegressor

        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (1.5 * X[:, 0] - 0.3 * X[:, 1]).astype(np.float32)
        model = PerpetualRegressor(seed=0)
        model.fit(X, y)

        onx = native_to_onnx(model, X)
        self.assertIn("TreeEnsembleRegressor", [n.op_type for n in onx.proto.graph.node])

        expected = model.predict(X).astype(np.float32).reshape((-1, 1))

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, got, atol=1e-5)

        sess = self.check_ort(onx)
        ort = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort, atol=1e-5)

    def check_predictions(self, model, X, target_opset=None):
        """Checks native and ORT predictions on dynamic input batches."""
        kwargs = {} if target_opset is None else {"target_opset": target_opset}
        onx = native_to_onnx(model, X[:1], **kwargs)
        expected = model.predict(X).astype(np.float32).reshape((-1, 1))
        self.assertEqualArray(
            expected, ExtendedReferenceEvaluator(onx).run(None, {"X": X})[0], atol=1e-5
        )
        self.assertEqualArray(expected, self.check_ort(onx).run(None, {"X": X})[0], atol=1e-5)

    def test_missing_values_and_boundaries(self):
        from perpetual import PerpetualRegressor

        for dtype in (np.float32, np.float64):
            for separate_missing in (False, True):
                with self.subTest(dtype=dtype, separate_missing=separate_missing):
                    rng = np.random.default_rng(12)
                    X = rng.normal(size=(80, 3)).astype(dtype)
                    X[::5, 0] = np.nan
                    y = np.where(np.isnan(X[:, 0]), 8, X[:, 0] * 2 + X[:, 1])
                    model = PerpetualRegressor(
                        seed=0, budget=0.1, num_threads=1, create_missing_branch=separate_missing
                    ).fit(X, y)
                    probes = [X, rng.normal(size=(15, 3)).astype(dtype)]
                    dump = json.loads(model.json_dump())
                    for tree in dump["trees"][:2]:
                        for node in tree["nodes"].values():
                            if node["is_leaf"] or node["split_value"] is None:
                                continue
                            threshold = dtype(node["split_value"])
                            rows = np.zeros((3, 3), dtype=dtype)
                            rows[:, node["split_feature"]] = [
                                np.nextafter(threshold, dtype(-np.inf)),
                                threshold,
                                np.nextafter(threshold, dtype(np.inf)),
                            ]
                            probes.append(rows)
                    self.check_predictions(model, np.concatenate(probes))

    def test_finite_missing_sentinel(self):
        from perpetual import PerpetualRegressor

        rng = np.random.default_rng(5)
        X = rng.normal(size=(60, 3)).astype(np.float32)
        X[::4, 1] = -999
        y = np.where(X[:, 1] == -999, 4, X[:, 1]).astype(np.float32)
        model = PerpetualRegressor(
            missing=-999, create_missing_branch=True, budget=0.1, num_threads=1
        ).fit(X, y)
        self.check_predictions(model, X)

    def test_constant_and_legacy_opset(self):
        from perpetual import PerpetualRegressor

        X = np.ones((20, 3), dtype=np.float32)
        model = PerpetualRegressor(budget=0.1, num_threads=1).fit(
            X, np.full(20, 2.5, dtype=np.float32)
        )
        for ml_opset in (1, 2, 3):
            with self.subTest(ml_opset=ml_opset):
                self.check_predictions(model, X, {"": 18, "ai.onnx.ml": ml_opset})

    def test_integer_features_and_pipeline(self):
        from perpetual import PerpetualRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.random.default_rng(8).integers(-10, 10, size=(60, 3))
        y = (X[:, 0] * 1.5 + X[:, 2]).astype(np.float64)
        model = PerpetualRegressor(budget=0.1, num_threads=1).fit(X, y)
        self.check_predictions(model, X)
        pipeline = make_pipeline(
            StandardScaler(), PerpetualRegressor(budget=0.1, num_threads=1)
        ).fit(X.astype(np.float32), y)
        onx = native_to_onnx(pipeline, X.astype(np.float32))
        # Perpetual 2.1 lacks the sklearn tags required by Pipeline.predict.
        expected = (
            pipeline[-1]
            .predict(pipeline[0].transform(X.astype(np.float32)))
            .astype(np.float32)
            .reshape((-1, 1))
        )
        for evaluator in (ExtendedReferenceEvaluator(onx), self.check_ort(onx)):
            self.assertEqualArray(
                expected, evaluator.run(None, {"X": X.astype(np.float32)})[0], atol=1e-5
            )

    def test_unsupported_categorical_splits(self):
        from perpetual import PerpetualRegressor

        rng = np.random.default_rng(4)
        X = rng.normal(size=(30, 2)).astype(np.float32)
        model = PerpetualRegressor(budget=0.1, num_threads=1).fit(X, X[:, 0].copy())
        dump = json.loads(model.json_dump())
        branch = next(
            node
            for tree in dump["trees"]
            for node in tree["nodes"].values()
            if not node["is_leaf"]
        )
        branch["left_cats"] = [1]
        with (
            patch.object(model, "json_dump", return_value=json.dumps(dump)),
            self.assertRaisesRegex(NotImplementedError, "categorical splits"),
        ):
            native_to_onnx(model, X)
        branch["left_cats"] = None
        branch["split_value"] = None
        with (
            patch.object(model, "json_dump", return_value=json.dumps(dump)),
            self.assertRaisesRegex(NotImplementedError, "non-finite split thresholds"),
        ):
            native_to_onnx(model, X)
        model.cat_mapping = {"feature": {"category": 0}}
        with self.assertRaisesRegex(NotImplementedError, "categorical input mappings"):
            native_to_onnx(model, X)

    def test_not_fitted(self):
        from perpetual import PerpetualRegressor

        with self.assertRaisesRegex(ValueError, "must be fitted"):
            native_to_onnx(PerpetualRegressor(), np.zeros((2, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
