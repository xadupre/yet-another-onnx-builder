import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_perpetual, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from unittests.sklearn._perpetual._helpers import native_to_onnx


@requires_sklearn("1.4")
@requires_perpetual("2.1")
class TestPerpetualClassifier(ExtTestCase):
    def test_perpetual_classifier(self):
        from perpetual import PerpetualClassifier

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
        model = PerpetualClassifier(seed=0)
        model.fit(X, y)

        onx = native_to_onnx(model, X)
        self.assertIn("TreeEnsembleRegressor", [n.op_type for n in onx.proto.graph.node])

        expected_label = model.predict(X)
        expected_proba = model.predict_proba(X).astype(np.float32)

        ref = ExtendedReferenceEvaluator(onx)
        got_label, got_proba = ref.run(None, {"X": X})
        self.assertEqualArray(expected_label, got_label)
        self.assertEqualArray(expected_proba, got_proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_label, ort_proba = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_label)
        self.assertEqualArray(expected_proba, ort_proba, atol=1e-5)

    def check_predictions(self, model, X):
        """Checks labels and probabilities with native and ORT evaluators."""
        onx = native_to_onnx(model, X[:1])
        expected_label = model.predict(X)
        expected_proba = model.predict_proba(X).astype(np.float32)
        for evaluator in (ExtendedReferenceEvaluator(onx), self.check_ort(onx)):
            label, proba = evaluator.run(None, {"X": X})
            np.testing.assert_array_equal(expected_label, label)
            self.assertEqualArray(expected_proba, proba, atol=1e-5)
            self.assertEqualArray(np.ones(X.shape[0], dtype=np.float32), proba.sum(axis=1))

    def test_multiclass_labels(self):
        from perpetual import PerpetualClassifier

        rng = np.random.default_rng(6)
        X = rng.normal(size=(90, 4)).astype(np.float32)
        indices = np.argmax(X[:, :3], axis=1)
        for classes in (
            np.array([-4, 2, 9], dtype=np.int64),
            np.array(["blue", "green", "red"]),
            np.array([-1.5, 2.5, 8.5]),
        ):
            with self.subTest(classes=classes):
                model = PerpetualClassifier(seed=0, budget=0.1, num_threads=1).fit(
                    X, classes[indices]
                )
                self.check_predictions(model, np.concatenate([X, X[:7] + 0.05]))

    def test_binary_encoded_labels_and_missing(self):
        from perpetual import PerpetualClassifier

        rng = np.random.default_rng(7)
        X = rng.normal(size=(70, 3))
        y = np.where(X[:, 0] > 0, "yes", "no")
        X[::5, 0] = np.nan
        model = PerpetualClassifier(
            seed=0, budget=0.1, num_threads=1, create_missing_branch=True
        ).fit(X, y)
        self.check_predictions(model, X)

    def test_constant_binary_tie_and_multiclass(self):
        from perpetual import PerpetualClassifier

        X = np.zeros((60, 3), dtype=np.float32)
        for n_classes in (2, 3):
            with self.subTest(n_classes=n_classes):
                model = PerpetualClassifier(budget=0.1, num_threads=1).fit(
                    X, np.arange(60) % n_classes
                )
                self.check_predictions(model, X)


if __name__ == "__main__":
    unittest.main(verbosity=2)
