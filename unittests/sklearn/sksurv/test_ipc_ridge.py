"""
Unit tests for the IPCRidge converter.
"""

import unittest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_sksurv
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


def _make_survival_data(n=50, n_features=5, seed=0):
    """Return (X, y) with a structured-array survival target."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    time = rng.exponential(scale=10, size=n)
    event = rng.choice([True, False], size=n)
    y = np.array(
        [(e, t) for e, t in zip(event, time)],
        dtype=[("event", "?"), ("time", "f8")],
    )
    return X, y


@requires_sklearn("1.4")
@requires_sksurv()
class TestSklearnIPCRidge(ExtTestCase):
    _X, _y = _make_survival_data()

    def _check_regressor(self, X, y, atol=1e-5, **kwargs):
        from sksurv.linear_model import IPCRidge

        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            reg = IPCRidge(**kwargs)
            reg.fit(Xd, y)
            onx = to_onnx(reg, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            predictions = results[0]

            expected = reg.predict(Xd).astype(dtype).reshape(-1, 1)
            self.assertEqualArray(expected, predictions, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected, ort_results[0], atol=atol)

    def test_regressor_default(self):
        """IPCRidge with default parameters."""
        self._check_regressor(self._X, self._y)

    def test_regressor_alpha(self):
        """IPCRidge with a custom regularisation strength."""
        self._check_regressor(self._X, self._y, alpha=0.5)

    def test_regressor_in_pipeline(self):
        """IPCRidge as last step in a sklearn Pipeline."""
        from sksurv.linear_model import IPCRidge

        Xd = self._X.astype(np.float32)
        reg = IPCRidge(alpha=1.0)
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
        pipe.fit(Xd, self._y)

        onx = to_onnx(pipe, (Xd,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": Xd})
        expected = pipe.predict(Xd).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, results[0], atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": Xd})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
