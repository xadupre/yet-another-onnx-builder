import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_perpetual, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


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

        onx = to_onnx(model, (X,))
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
