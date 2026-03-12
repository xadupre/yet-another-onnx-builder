"""
Unit tests for yobx.sklearn.cluster.AffinityPropagation converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestAffinityPropagation(ExtTestCase):
    def test_affinity_propagation_labels(self):
        from sklearn.cluster import AffinityPropagation
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        ap = AffinityPropagation(random_state=0)
        ap.fit(X)

        onx = to_onnx(ap, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("ArgMin", op_types)
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = ap.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_affinity_propagation_distances(self):
        from sklearn.cluster import AffinityPropagation
        from sklearn.metrics import pairwise_distances
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        ap = AffinityPropagation(random_state=1)
        ap.fit(X)

        onx = to_onnx(ap, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        _labels, distances = results[0], results[1]

        # Distances should match Euclidean distances to cluster centers.
        expected_distances = pairwise_distances(X, ap.cluster_centers_.astype(np.float32)).astype(
            np.float32
        )
        self.assertEqualArray(expected_distances, distances, atol=1e-3)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_distances, ort_results[1], atol=1e-3)

    def test_affinity_propagation_damping(self):
        from sklearn.cluster import AffinityPropagation
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 2)).astype(np.float32)
        ap = AffinityPropagation(damping=0.7, random_state=0)
        ap.fit(X)

        onx = to_onnx(ap, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = ap.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

    def test_affinity_propagation_pipeline(self):
        from sklearn.cluster import AffinityPropagation
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ap", AffinityPropagation(random_state=0)),
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
