import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_perpetual, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


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

        onx = to_onnx(model, (X,))
        self.assertIn("TreeEnsembleRegressor", [n.op_type for n in onx.proto.graph.node])

        expected = model.predict(X).astype(np.float32).reshape((-1, 1))

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, got, atol=1e-5)

        sess = self.check_ort(onx)
        ort = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
