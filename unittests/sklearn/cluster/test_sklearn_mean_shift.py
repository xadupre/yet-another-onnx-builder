"""
Unit tests for yobx.sklearn.cluster.MeanShift converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestMeanShift(ExtTestCase):
    def test_mean_shift_labels(self):
        from sklearn.cluster import MeanShift
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        ms = MeanShift()
        ms.fit(X)

        onx = to_onnx(ms, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("ArgMin", op_types)
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = ms.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_mean_shift_distances(self):
        from sklearn.cluster import MeanShift
        from sklearn.metrics import pairwise_distances
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        ms = MeanShift()
        ms.fit(X)

        onx = to_onnx(ms, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        _labels, distances = results[0], results[1]

        # Distances should match Euclidean distances to cluster centers.
        expected_distances = pairwise_distances(
            X, ms.cluster_centers_.astype(np.float32)
        ).astype(np.float32)
        self.assertEqualArray(expected_distances, distances, atol=1e-3)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_distances, ort_results[1], atol=1e-3)

    def test_mean_shift_bandwidth(self):
        from sklearn.cluster import MeanShift
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 2)).astype(np.float32)
        ms = MeanShift(bandwidth=1.5)
        ms.fit(X)

        onx = to_onnx(ms, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = ms.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

    def test_mean_shift_float64(self):
        from sklearn.cluster import MeanShift
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 3)).astype(np.float64)
        ms = MeanShift()
        ms.fit(X)

        onx = to_onnx(ms, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = ms.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_mean_shift_pipeline(self):
        from sklearn.cluster import MeanShift
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ms", MeanShift()),
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
