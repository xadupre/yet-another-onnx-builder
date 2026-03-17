"""
Unit tests for yobx.sklearn.kernel_approximation.additive_chi2_sampler converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.0")
class TestAdditiveChi2Sampler(ExtTestCase):
    def _make_data(self, seed=0, n_samples=30, n_features=5):
        rng = np.random.default_rng(seed)
        # AdditiveChi2Sampler requires non-negative input.
        return np.abs(rng.standard_normal((n_samples, n_features))).astype(np.float32)

    def _run(self, sample_steps, sample_interval=None, dtype=np.float32):
        from sklearn.kernel_approximation import AdditiveChi2Sampler
        from yobx.sklearn import to_onnx

        X = self._make_data().astype(dtype)
        est = AdditiveChi2Sampler(
            sample_steps=sample_steps,
            **({"sample_interval": sample_interval} if sample_interval is not None else {}),
        )
        est.fit(X)

        onx = to_onnx(est, (X,))

        expected = est.transform(X).astype(dtype)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_sample_steps_1(self):
        self._run(sample_steps=1)

    def test_sample_steps_2(self):
        self._run(sample_steps=2)

    def test_sample_steps_3(self):
        self._run(sample_steps=3)

    def test_custom_sample_interval(self):
        self._run(sample_steps=4, sample_interval=0.3)

    def test_float64(self):
        # ORT does not implement Cos/Sin for float64, so we only verify
        # correctness via the reference evaluator, not via ORT.
        from sklearn.kernel_approximation import AdditiveChi2Sampler
        from yobx.sklearn import to_onnx

        X = self._make_data().astype(np.float64)
        est = AdditiveChi2Sampler(sample_steps=2)
        est.fit(X)

        onx = to_onnx(est, (X,))

        expected = est.transform(X)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-10)

    def test_zeros_in_input(self):
        """Zero-valued inputs must produce zero outputs for every component."""
        from sklearn.kernel_approximation import AdditiveChi2Sampler
        from yobx.sklearn import to_onnx

        # Create a matrix that includes exact zeros.
        X = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
        est = AdditiveChi2Sampler(sample_steps=2)
        est.fit(X)

        onx = to_onnx(est, (X,))

        expected = est.transform(X).astype(np.float32)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_out_of_sample(self):
        """Verify the converter works on data not seen during fitting."""
        from sklearn.kernel_approximation import AdditiveChi2Sampler
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(42)
        X_train = np.abs(rng.standard_normal((20, 4))).astype(np.float32)
        X_test = np.abs(rng.standard_normal((8, 4))).astype(np.float32)

        est = AdditiveChi2Sampler(sample_steps=2)
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        expected = est.transform(X_test).astype(np.float32)
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
