"""
Unit tests for yobx.sklearn.neighbors RadiusNeighborsTransformer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestRadiusNeighborsTransformer(ExtTestCase):
    def test_rnn_transformer_distance_basic(self):
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((20, 4)).astype(np.float32)
        X_new = rng.standard_normal((8, 4)).astype(np.float32)

        est = RadiusNeighborsTransformer(radius=2.0, mode="distance")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Greater", op_types)
        self.assertIn("Where", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_new})
        output = results[0]

        expected = est.transform(X_new).toarray().astype(np.float32)
        self.assertEqualArray(expected, output, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_rnn_transformer_connectivity_basic(self):
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 4)).astype(np.float32)

        est = RadiusNeighborsTransformer(radius=2.0, mode="connectivity")
        est.fit(X)

        onx = to_onnx(est, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Cast", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        output = results[0]

        expected = est.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, output, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_rnn_transformer_distance_float64(self):
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X_train = rng.standard_normal((20, 4)).astype(np.float64)
        X_new = rng.standard_normal((6, 4)).astype(np.float64)

        est = RadiusNeighborsTransformer(radius=2.0, mode="distance")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_new})
        output = results[0]

        expected = est.transform(X_new).toarray().astype(np.float64)
        self.assertEqualArray(expected, output, atol=1e-10)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})
        self.assertEqualArray(expected, ort_results[0], atol=1e-10)

    def test_rnn_transformer_connectivity_float64(self):
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((20, 4)).astype(np.float64)
        X_new = rng.standard_normal((6, 4)).astype(np.float64)

        est = RadiusNeighborsTransformer(radius=2.0, mode="connectivity")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_new})
        output = results[0]

        expected = est.transform(X_new).toarray().astype(np.float64)
        self.assertEqualArray(expected, output, atol=1e-10)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})
        self.assertEqualArray(expected, ort_results[0], atol=1e-10)

    def test_rnn_transformer_output_shape(self):
        """Output shape should be (N_query, M_train)."""
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X_train = rng.standard_normal((30, 5)).astype(np.float32)
        X_new = rng.standard_normal((10, 5)).astype(np.float32)

        for mode in ["connectivity", "distance"]:
            with self.subTest(mode=mode):
                est = RadiusNeighborsTransformer(radius=2.0, mode=mode)
                est.fit(X_train)

                onx = to_onnx(est, (X_train,))

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X_new})
                self.assertEqual(ort_results[0].shape, (10, 30))

    def test_rnn_transformer_connectivity_values(self):
        """Connectivity output should be 0.0 or 1.0."""
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((25, 3)).astype(np.float32)

        est = RadiusNeighborsTransformer(radius=2.0, mode="connectivity")
        est.fit(X)

        onx = to_onnx(est, (X,))

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        output = ort_results[0]

        # All values should be 0.0 or 1.0
        unique_vals = np.unique(output)
        self.assertIn(0.0, unique_vals)
        self.assertIn(1.0, unique_vals)
        self.assertTrue(set(unique_vals.tolist()).issubset({0.0, 1.0}))

    def test_rnn_transformer_distance_nonneg(self):
        """Distance output values should be non-negative."""
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X_train = rng.standard_normal((25, 3)).astype(np.float32)
        X_new = rng.standard_normal((8, 3)).astype(np.float32)

        est = RadiusNeighborsTransformer(radius=2.0, mode="distance")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))
        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})
        output = ort_results[0]

        self.assertTrue(np.all(output >= 0.0))

    def test_rnn_transformer_com_microsoft(self):
        """CDist path (com.microsoft) should produce the same output."""
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(7)
        X_train = rng.standard_normal((20, 4)).astype(np.float32)
        X_new = rng.standard_normal((5, 4)).astype(np.float32)

        est = RadiusNeighborsTransformer(radius=2.0, mode="distance")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,), target_opset={"": 18, "com.microsoft": 1})

        op_types_domains = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types_domains)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})
        expected = est.transform(X_new).toarray().astype(np.float32)
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_rnn_transformer_metrics(self):
        """Test various distance metrics."""
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(8)
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
                est = RadiusNeighborsTransformer(
                    radius=2.0, mode="distance", metric=metric, **kwargs
                )
                est.fit(X_train)

                onx = to_onnx(est, (X_train,))
                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X_new})
                output = ort_results[0]

                self.assertEqual(output.shape, (10, 30))
                expected = est.transform(X_new).toarray().astype(np.float32)
                self.assertEqualArray(expected, output, atol=1e-5)

    def test_rnn_transformer_opset_too_low_raises(self):
        """Opset < 13 must raise NotImplementedError."""
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(9)
        X = rng.standard_normal((20, 3)).astype(np.float32)

        est = RadiusNeighborsTransformer(radius=2.0, mode="distance")
        est.fit(X)

        with self.assertRaises(NotImplementedError):
            to_onnx(est, (X,), target_opset=12)

    def test_rnn_transformer_small_radius(self):
        """With a very small radius, most entries should be zero."""
        from sklearn.neighbors import RadiusNeighborsTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(10)
        X_train = rng.standard_normal((20, 4)).astype(np.float32) * 10
        X_new = rng.standard_normal((5, 4)).astype(np.float32) * 10

        est = RadiusNeighborsTransformer(radius=0.001, mode="distance")
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_new})
        output = ort_results[0]

        expected = est.transform(X_new).toarray().astype(np.float32)
        self.assertEqualArray(expected, output, atol=1e-5)

        # With very small radius, most entries are 0
        self.assertTrue(np.sum(output == 0) > np.sum(output != 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
