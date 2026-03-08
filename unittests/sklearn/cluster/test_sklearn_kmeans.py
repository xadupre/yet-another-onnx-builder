"""
Unit tests for yobx.sklearn.cluster.KMeans converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestKMeans(ExtTestCase):
    def test_kmeans_labels(self):
        from sklearn.cluster import KMeans
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        km = KMeans(n_clusters=3, random_state=0, n_init=10)
        km.fit(X)

        onx = to_onnx(km, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("ArgMin", op_types)
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels, _distances = results[0], results[1]

        expected_labels = km.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_kmeans_distances(self):
        from sklearn.cluster import KMeans
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        km = KMeans(n_clusters=4, random_state=1, n_init=10)
        km.fit(X)

        onx = to_onnx(km, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        _labels, distances = results[0], results[1]

        # Distances should match sklearn's transform output.
        expected_distances = km.transform(X).astype(np.float32)
        self.assertEqualArray(expected_distances, distances, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_distances, ort_results[1], atol=1e-4)

    def test_kmeans_two_clusters(self):
        from sklearn.cluster import KMeans
        from yobx.sklearn import to_onnx

        X = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        km = KMeans(n_clusters=2, random_state=0, n_init=10)
        km.fit(X)

        onx = to_onnx(km, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = km.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

    def test_kmeans_pipeline(self):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("km", KMeans(n_clusters=3, random_state=0, n_init=10)),
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
