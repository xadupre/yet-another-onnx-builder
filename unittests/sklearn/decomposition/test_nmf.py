"""
Unit tests for yobx.sklearn.decomposition.NMF and MiniBatchNMF converters.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestNMF(ExtTestCase):
    def _make_X(self, n_samples=20, n_features=6, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        return np.abs(rng.standard_normal((n_samples, n_features))).astype(np.float32)

    # ------------------------------------------------------------------
    # NMF (solver='mu')
    # ------------------------------------------------------------------

    def test_nmf_mu_basic(self):
        from sklearn.decomposition import NMF
        from yobx.sklearn import to_onnx

        X = self._make_X()
        nmf = NMF(n_components=3, solver="mu", max_iter=200, random_state=0, tol=0.0)
        nmf.fit(X)

        onx = to_onnx(nmf, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Loop", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = nmf.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_nmf_mu_float64(self):
        from sklearn.decomposition import NMF
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = np.abs(rng.standard_normal((20, 6))).astype(np.float64)
        nmf = NMF(n_components=2, solver="mu", max_iter=200, random_state=0, tol=0.0)
        nmf.fit(X)

        onx = to_onnx(nmf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = nmf.transform(X)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_nmf_mu_all_components(self):
        """n_components = n_features (square case)."""
        from sklearn.decomposition import NMF
        from yobx.sklearn import to_onnx

        X = self._make_X(n_features=4, rng_seed=2)
        nmf = NMF(n_components=4, solver="mu", max_iter=200, random_state=0, tol=0.0)
        nmf.fit(X)

        onx = to_onnx(nmf, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = nmf.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

    def test_nmf_mu_single_component(self):
        from sklearn.decomposition import NMF
        from yobx.sklearn import to_onnx

        X = self._make_X(rng_seed=3)
        nmf = NMF(n_components=1, solver="mu", max_iter=200, random_state=0, tol=0.0)
        nmf.fit(X)

        onx = to_onnx(nmf, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = nmf.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

    def test_nmf_cd_raises(self):
        """solver='cd' must raise NotImplementedError."""
        from sklearn.decomposition import NMF
        from yobx.sklearn import to_onnx

        X = self._make_X()
        nmf = NMF(n_components=2, solver="cd", max_iter=50, random_state=0)
        nmf.fit(X)

        with self.assertRaises(NotImplementedError):
            to_onnx(nmf, (X,))

    def test_nmf_pipeline(self):
        from sklearn.decomposition import NMF
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = self._make_X()
        y = X[:, 0]  # regression target

        pipe = Pipeline(
            [
                ("nmf", NMF(n_components=3, solver="mu", max_iter=100, random_state=0, tol=0.0)),
                ("reg", Ridge()),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0].ravel()
        expected = pipe.predict(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-3)

    # ------------------------------------------------------------------
    # MiniBatchNMF
    # ------------------------------------------------------------------

    def test_mini_batch_nmf_basic(self):
        from sklearn.decomposition import MiniBatchNMF
        from yobx.sklearn import to_onnx

        X = self._make_X()
        nmf = MiniBatchNMF(n_components=3, random_state=0, tol=0.0, max_no_improvement=None)
        nmf.fit(X)

        onx = to_onnx(nmf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = nmf.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_mini_batch_nmf_float64(self):
        from sklearn.decomposition import MiniBatchNMF
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = np.abs(rng.standard_normal((20, 6))).astype(np.float64)
        nmf = MiniBatchNMF(n_components=2, random_state=0, tol=0.0, max_no_improvement=None)
        nmf.fit(X)

        onx = to_onnx(nmf, (X,))
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = nmf.transform(X)
        self.assertEqualArray(expected, result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
