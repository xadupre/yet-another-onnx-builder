"""
Unit tests for yobx.sklearn.neighbors KNeighborsTransformer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestKNeighborsTransformer(ExtTestCase):
    def test_knn_transformer_distance_basic(self):
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((20, 4)).astype(np.float32)
        X_new = rng.standard_normal((8, 4)).astype(np.float32)

        est = KNeighborsTransformer(n_neighbors=3, mode="distance")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TopK", op_types)
        self.assertIn("ScatterElements", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_new})
        output = results[0]

        # Build expected from kneighbors() using new data (no self-connection).
        dists, inds = est.kneighbors(X_new, n_neighbors=3)
        expected = np.zeros((8, 20), dtype=np.float32)
        for i in range(8):
            expected[i, inds[i]] = dists[i]
        self.assertEqualArray(expected, output, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_knn_transformer_connectivity_basic(self):
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 4)).astype(np.float32)

        est = KNeighborsTransformer(n_neighbors=3, mode="connectivity")
        est.fit(X)

        onx = to_onnx(est, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        output = results[0]

        sklearn_out = est.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(sklearn_out, output, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(sklearn_out, ort_results[0], atol=1e-5)

    def test_knn_transformer_connectivity_float32(self):
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(9)
        X_train = rng.standard_normal((20, 4)).astype(np.float32)
        X_new = rng.standard_normal((8, 4)).astype(np.float32)

        est = KNeighborsTransformer(n_neighbors=3, mode="connectivity")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TopK", op_types)
        self.assertIn("ScatterElements", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_new})
        output = results[0]

        # Build expected: 1.0 at k-NN positions, 0.0 elsewhere
        _, inds = est.kneighbors(X_new, n_neighbors=3)
        expected = np.zeros((8, 20), dtype=np.float32)
        for i in range(8):
            expected[i, inds[i]] = 1.0
        self.assertEqualArray(expected, output, atol=1e-7)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})
        self.assertEqualArray(expected, ort_results[0], atol=1e-7)

    def test_knn_transformer_connectivity_float64(self):
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(10)
        X_train = rng.standard_normal((20, 4)).astype(np.float64)
        X_new = rng.standard_normal((6, 4)).astype(np.float64)

        est = KNeighborsTransformer(n_neighbors=3, mode="connectivity")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_new})
        output = results[0]

        # Build expected: 1.0 at k-NN positions, 0.0 elsewhere
        _, inds = est.kneighbors(X_new, n_neighbors=3)
        expected = np.zeros((6, 20), dtype=np.float64)
        for i in range(6):
            expected[i, inds[i]] = 1.0
        self.assertEqualArray(expected, output, atol=1e-10)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})
        self.assertEqualArray(expected, ort_results[0], atol=1e-10)

    def test_knn_transformer_distance_float64(self):
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X_train = rng.standard_normal((20, 4)).astype(np.float64)
        X_new = rng.standard_normal((6, 4)).astype(np.float64)

        est = KNeighborsTransformer(n_neighbors=3, mode="distance")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_new})
        output = results[0]

        # Build expected from kneighbors() using new data (no self-connection)
        dists, inds = est.kneighbors(X_new, n_neighbors=3)
        expected = np.zeros((6, 20), dtype=np.float64)
        for i in range(6):
            expected[i, inds[i]] = dists[i]
        self.assertEqualArray(expected, output, atol=1e-10)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})
        self.assertEqualArray(expected, ort_results[0], atol=1e-10)

    def test_knn_transformer_output_shape(self):
        """Output shape should be (N_query, M_train)."""
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((30, 5)).astype(np.float32)
        X_new = rng.standard_normal((10, 5)).astype(np.float32)

        for mode in ["connectivity", "distance"]:
            with self.subTest(mode=mode):
                est = KNeighborsTransformer(n_neighbors=4, mode=mode)
                est.fit(X_train)

                onx = to_onnx(est, (X_train,))

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X_new})
                self.assertEqual(ort_results[0].shape, (10, 30))

    def test_knn_transformer_connectivity_values(self):
        """Connectivity output should be 0.0 or 1.0."""
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X = rng.standard_normal((25, 3)).astype(np.float32)

        est = KNeighborsTransformer(n_neighbors=4, mode="connectivity")
        est.fit(X)

        onx = to_onnx(est, (X,))

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        output = ort_results[0]

        # All values should be 0.0 or 1.0
        unique_vals = np.unique(output)
        self.assertIn(0.0, unique_vals)
        self.assertIn(1.0, unique_vals)
        self.assertEqual(set(unique_vals.tolist()), {0.0, 1.0})

        # Each row should have exactly n_neighbors non-zero entries
        nonzero_per_row = (output != 0).sum(axis=1)
        np.testing.assert_array_equal(nonzero_per_row, 4)

    def test_knn_transformer_distance_nonzero_per_row(self):
        """Distance output should have exactly n_neighbors non-zero entries per row."""
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X_train = rng.standard_normal((25, 3)).astype(np.float32)
        X_new = rng.standard_normal((8, 3)).astype(np.float32)

        for k in [3, 5, 7]:
            with self.subTest(k=k):
                est = KNeighborsTransformer(n_neighbors=k, mode="distance")
                est.fit(X_train)

                onx = to_onnx(est, (X_train,))
                sess = self.check_ort(onx)
                # Use new data so no self-distance precision issues
                ort_results = sess.run(None, {"X": X_new})
                output = ort_results[0]

                nonzero_per_row = (output != 0).sum(axis=1)
                np.testing.assert_array_equal(nonzero_per_row, k)

    def test_knn_transformer_com_microsoft(self):
        """CDist path (com.microsoft) should produce the same output."""
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X_train = rng.standard_normal((20, 4)).astype(np.float32)
        X_new = rng.standard_normal((5, 4)).astype(np.float32)

        est = KNeighborsTransformer(n_neighbors=3, mode="distance")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,), target_opset={"": 18, "com.microsoft": 1})

        op_types_domains = [(n.op_type, n.domain) for n in onx.proto.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types_domains)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})

        # Build expected output from kneighbors() using new data (no self-connection)
        dists, inds = est.kneighbors(X_new, n_neighbors=3)
        expected = np.zeros((5, 20), dtype=np.float32)
        for i in range(5):
            expected[i, inds[i]] = dists[i]
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_knn_transformer_metrics(self):
        """Test various distance metrics."""
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(7)
        X_train = rng.standard_normal((30, 4)).astype(np.float32)
        X_new = rng.standard_normal((10, 4)).astype(np.float32)

        metrics = [
            ("euclidean", {}),
            ("cosine", {}),
            ("manhattan", {}),
            ("chebyshev", {}),
            ("minkowski", {"p": 3}),
        ]
        for metric, kwargs in metrics:
            with self.subTest(metric=metric, **kwargs):
                est = KNeighborsTransformer(
                    n_neighbors=4, mode="distance", metric=metric, **kwargs
                )
                est.fit(X_train)

                onx = to_onnx(est, (X_train,))
                sess = self.check_ort(onx)
                # Use new data to avoid self-distance precision issues
                ort_results = sess.run(None, {"X": X_new})
                output = ort_results[0]

                self.assertEqual(output.shape, (10, 30))
                # Each row should have exactly n_neighbors non-zero entries
                nonzero_per_row = (output != 0).sum(axis=1)
                np.testing.assert_array_equal(nonzero_per_row, 4)

    def test_knn_transformer_opset_too_low_raises(self):
        """Opset < 13 must raise NotImplementedError."""
        from sklearn.neighbors import KNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(8)
        X = rng.standard_normal((20, 3)).astype(np.float32)

        est = KNeighborsTransformer(n_neighbors=3, mode="distance")
        est.fit(X)

        with self.assertRaises(NotImplementedError):
            to_onnx(est, (X,), target_opset=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
