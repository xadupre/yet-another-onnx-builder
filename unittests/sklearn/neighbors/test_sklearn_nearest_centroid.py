"""
Unit tests for yobx.sklearn.neighbors NearestCentroid converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.8")
class TestNearestCentroid(ExtTestCase):
    def test_nearest_centroid_basic(self):
        from sklearn.neighbors import NearestCentroid
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = NearestCentroid()
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("ArgMin", op_types)
        self.assertIn("Gather", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels, probabilities = results[0], results[1]

        expected_labels = clf.predict(X).astype(np.int64)
        expected_proba = clf.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_labels, labels)
        self.assertEqualArray(expected_proba, probabilities, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_nearest_centroid_float64(self):
        from sklearn.neighbors import NearestCentroid
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 4)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = NearestCentroid()
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0])

    def test_nearest_centroid_manhattan(self):
        from sklearn.neighbors import NearestCentroid
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = NearestCentroid(metric="manhattan")
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0])

    def test_nearest_centroid_multiclass(self):
        from sklearn.neighbors import NearestCentroid
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] * 3).astype(np.int64) % 3  # 3 classes: 0, 1, 2

        clf = NearestCentroid()
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0])

    def test_nearest_centroid_non_uniform_prior(self):
        """Test NearestCentroid with non-uniform class priors (empirical priors)."""
        from sklearn.neighbors import NearestCentroid
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        # Create an imbalanced dataset to trigger non-uniform priors.
        X_pos = rng.standard_normal((30, 4)).astype(np.float32) + 2.0
        X_neg = rng.standard_normal((10, 4)).astype(np.float32) - 2.0
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * 30 + [0] * 10, dtype=np.int64)

        clf = NearestCentroid(priors="empirical")
        clf.fit(X, y)

        # Only run this test if priors are actually non-uniform.
        if np.allclose(clf.class_prior_, clf.class_prior_[0]):
            return  # pragma: no cover

        onx = to_onnx(clf, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("ArgMax", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0])

    def test_nearest_centroid_shrink_threshold(self):
        """Test NearestCentroid with shrink_threshold."""
        from sklearn.neighbors import NearestCentroid
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((40, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = NearestCentroid(shrink_threshold=0.5)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
