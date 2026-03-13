"""
Unit tests for yobx.sklearn.cluster.BisectingKMeans converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestBisectingKMeans(ExtTestCase):
    # ------------------------------------------------------------------
    # Labels — float32 and float64
    # ------------------------------------------------------------------

    def test_bisecting_kmeans_labels_float32(self):
        from sklearn.cluster import BisectingKMeans
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        bkm = BisectingKMeans(n_clusters=3, random_state=0)
        bkm.fit(X)

        onx = to_onnx(bkm, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("LessOrEqual", op_types)
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = bkm.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_bisecting_kmeans_labels_float64(self):
        from sklearn.cluster import BisectingKMeans
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        bkm = BisectingKMeans(n_clusters=3, random_state=0)
        bkm.fit(X)

        onx = to_onnx(bkm, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = bkm.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    # ------------------------------------------------------------------
    # Distances — float32 and float64 (standard ONNX path, no CDist)
    # ------------------------------------------------------------------

    def test_bisecting_kmeans_distances_float32(self):
        from sklearn.cluster import BisectingKMeans
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        bkm = BisectingKMeans(n_clusters=4, random_state=1)
        bkm.fit(X)

        onx = to_onnx(bkm, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        _labels, distances = results[0], results[1]

        expected_distances = bkm.transform(X).astype(np.float32)
        self.assertEqualArray(expected_distances, distances, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_distances, ort_results[1], atol=1e-4)

    def test_bisecting_kmeans_distances_float64(self):
        from sklearn.cluster import BisectingKMeans
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 3)).astype(np.float64)
        bkm = BisectingKMeans(n_clusters=4, random_state=1)
        bkm.fit(X)

        onx = to_onnx(bkm, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        _labels, distances = results[0], results[1]

        expected_distances = bkm.transform(X).astype(np.float64)
        self.assertEqualArray(expected_distances, distances, atol=1e-6)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_distances, ort_results[1], atol=1e-6)

    # ------------------------------------------------------------------
    # CDist path (com.microsoft opset)
    # ------------------------------------------------------------------

    def test_bisecting_kmeans_cdist_float32(self):
        """Distances computed via com.microsoft.CDist (float32)."""
        from sklearn.cluster import BisectingKMeans
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        bkm = BisectingKMeans(n_clusters=3, random_state=0)
        bkm.fit(X)

        onx = to_onnx(bkm, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        domains = {oi.domain for oi in onx.opset_import}
        self.assertIn("com.microsoft", domains)

        expected_labels = bkm.predict(X).astype(np.int64)
        expected_distances = bkm.transform(X).astype(np.float32)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])
        self.assertEqualArray(expected_distances, ort_results[1], atol=1e-4)

    def test_bisecting_kmeans_cdist_float64(self):
        """Distances computed via com.microsoft.CDist (float64)."""
        from sklearn.cluster import BisectingKMeans
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        bkm = BisectingKMeans(n_clusters=3, random_state=0)
        bkm.fit(X)

        onx = to_onnx(bkm, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        expected_labels = bkm.predict(X).astype(np.int64)
        expected_distances = bkm.transform(X).astype(np.float64)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])
        self.assertEqualArray(expected_distances, ort_results[1], atol=1e-6)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_bisecting_kmeans_two_clusters(self):
        from sklearn.cluster import BisectingKMeans
        from yobx.sklearn import to_onnx

        X = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        bkm = BisectingKMeans(n_clusters=2, random_state=0)
        bkm.fit(X)

        onx = to_onnx(bkm, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = bkm.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def test_bisecting_kmeans_pipeline(self):
        from sklearn.cluster import BisectingKMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("bkm", BisectingKMeans(n_clusters=3, random_state=0)),
            ]
        )
        pipe.fit(X)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = pipe.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
