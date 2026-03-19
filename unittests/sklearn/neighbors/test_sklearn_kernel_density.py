"""
Unit tests for yobx.sklearn.neighbors.KernelDensity converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.2")
class TestKernelDensity(ExtTestCase):
    # ------------------------------------------------------------------
    # Gaussian kernel — float32 and float64
    # ------------------------------------------------------------------

    def test_kernel_density_gaussian_float32(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        kde = KernelDensity(bandwidth=1.0, kernel="gaussian")
        kde.fit(X)

        onx = to_onnx(kde, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("ReduceLogSumExp", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        log_density = results[0]

        expected = kde.score_samples(X).astype(np.float32)
        self.assertEqualArray(expected, log_density, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_kernel_density_gaussian_float64(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        kde = KernelDensity(bandwidth=1.0, kernel="gaussian")
        kde.fit(X)

        onx = to_onnx(kde, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        log_density = results[0]

        expected = kde.score_samples(X).astype(np.float64)
        self.assertEqualArray(expected, log_density, atol=1e-6)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-6)

    # ------------------------------------------------------------------
    # Tophat kernel — float32 and float64
    # ------------------------------------------------------------------

    def test_kernel_density_tophat_float32(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X_train = rng.standard_normal((40, 3)).astype(np.float32)
        X_test = rng.standard_normal((10, 3)).astype(np.float32)
        kde = KernelDensity(bandwidth=1.5, kernel="tophat")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        log_density = results[0]

        expected = kde.score_samples(X_test).astype(np.float32)
        self.assertEqualArray(expected, log_density, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_kernel_density_tophat_float64(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((40, 3)).astype(np.float64)
        X_test = rng.standard_normal((10, 3)).astype(np.float64)
        kde = KernelDensity(bandwidth=1.5, kernel="tophat")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        log_density = results[0]

        expected = kde.score_samples(X_test).astype(np.float64)
        self.assertEqualArray(expected, log_density, atol=1e-6)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-6)

    # ------------------------------------------------------------------
    # Epanechnikov kernel
    # ------------------------------------------------------------------

    def test_kernel_density_epanechnikov_float32(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X_train = rng.standard_normal((40, 3)).astype(np.float32)
        X_test = rng.standard_normal((10, 3)).astype(np.float32)
        kde = KernelDensity(bandwidth=2.0, kernel="epanechnikov")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        log_density = results[0]

        expected = kde.score_samples(X_test).astype(np.float32)
        self.assertEqualArray(expected, log_density, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_kernel_density_epanechnikov_float64(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X_train = rng.standard_normal((40, 3)).astype(np.float64)
        X_test = rng.standard_normal((10, 3)).astype(np.float64)
        kde = KernelDensity(bandwidth=2.0, kernel="epanechnikov")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        log_density = results[0]

        expected = kde.score_samples(X_test).astype(np.float64)
        self.assertEqualArray(expected, log_density, atol=1e-6)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-6)

    # ------------------------------------------------------------------
    # Exponential kernel
    # ------------------------------------------------------------------

    def test_kernel_density_exponential_float32(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X_train = rng.standard_normal((30, 4)).astype(np.float32)
        X_test = rng.standard_normal((8, 4)).astype(np.float32)
        kde = KernelDensity(bandwidth=1.0, kernel="exponential")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("ReduceLogSumExp", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        log_density = results[0]

        expected = kde.score_samples(X_test).astype(np.float32)
        self.assertEqualArray(expected, log_density, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_kernel_density_exponential_float64(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(7)
        X_train = rng.standard_normal((30, 4)).astype(np.float64)
        X_test = rng.standard_normal((8, 4)).astype(np.float64)
        kde = KernelDensity(bandwidth=1.0, kernel="exponential")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        log_density = results[0]

        expected = kde.score_samples(X_test).astype(np.float64)
        self.assertEqualArray(expected, log_density, atol=1e-6)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-6)

    # ------------------------------------------------------------------
    # Linear kernel
    # ------------------------------------------------------------------

    def test_kernel_density_linear_float32(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(8)
        X_train = rng.standard_normal((40, 3)).astype(np.float32)
        X_test = rng.standard_normal((10, 3)).astype(np.float32)
        kde = KernelDensity(bandwidth=2.0, kernel="linear")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        log_density = results[0]

        expected = kde.score_samples(X_test).astype(np.float32)
        self.assertEqualArray(expected, log_density, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_kernel_density_linear_float64(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(9)
        X_train = rng.standard_normal((40, 3)).astype(np.float64)
        X_test = rng.standard_normal((10, 3)).astype(np.float64)
        kde = KernelDensity(bandwidth=2.0, kernel="linear")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        log_density = results[0]

        expected = kde.score_samples(X_test).astype(np.float64)
        self.assertEqualArray(expected, log_density, atol=1e-6)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-6)

    # ------------------------------------------------------------------
    # CDist path (com.microsoft opset) — gaussian kernel
    # ------------------------------------------------------------------

    def test_kernel_density_cdist_float32(self):
        """Squared distances computed via com.microsoft.CDist (float32)."""
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(10)
        X_train = rng.standard_normal((30, 4)).astype(np.float32)
        X_test = rng.standard_normal((8, 4)).astype(np.float32)
        kde = KernelDensity(bandwidth=1.0, kernel="gaussian")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.proto.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        expected = kde.score_samples(X_test).astype(np.float32)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_kernel_density_cdist_float64(self):
        """Squared distances computed via com.microsoft.CDist (float64)."""
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(11)
        X_train = rng.standard_normal((30, 4)).astype(np.float64)
        X_test = rng.standard_normal((8, 4)).astype(np.float64)
        kde = KernelDensity(bandwidth=1.0, kernel="gaussian")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.proto.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        expected = kde.score_samples(X_test).astype(np.float64)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-6)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def test_kernel_density_pipeline(self):
        from sklearn.neighbors import KernelDensity
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(12)
        X_train = rng.standard_normal((40, 4)).astype(np.float32)
        X_test = rng.standard_normal((10, 4)).astype(np.float32)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("kde", KernelDensity(bandwidth=1.0, kernel="gaussian")),
            ]
        )
        pipe.fit(X_train)

        onx = to_onnx(pipe, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        log_density = results[0]

        expected = pipe.score_samples(X_test).astype(np.float32)
        self.assertEqualArray(expected, log_density, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    # ------------------------------------------------------------------
    # Different bandwidth
    # ------------------------------------------------------------------

    def test_kernel_density_bandwidth(self):
        from sklearn.neighbors import KernelDensity
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(13)
        X_train = rng.standard_normal((30, 2)).astype(np.float32)
        X_test = rng.standard_normal((5, 2)).astype(np.float32)
        kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
        kde.fit(X_train)

        onx = to_onnx(kde, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        log_density = results[0]

        expected = kde.score_samples(X_test).astype(np.float32)
        self.assertEqualArray(expected, log_density, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
