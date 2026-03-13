"""
Unit tests for yobx.sklearn.cluster.Birch converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestBirch(ExtTestCase):
    def test_birch_labels(self):
        from sklearn.cluster import Birch
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        model = Birch(n_clusters=3)
        model.fit(X)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("ArgMin", op_types)
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels, _distances = results[0], results[1]

        expected_labels = model.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_birch_distances(self):
        from sklearn.cluster import Birch
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        model = Birch(n_clusters=4)
        model.fit(X)

        onx = to_onnx(model, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        _labels, distances = results[0], results[1]

        # Distances should match Euclidean distances to subcluster centers.
        centers = model.subcluster_centers_.astype(np.float32)
        expected_distances = np.sqrt(
            np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        )
        self.assertEqualArray(expected_distances, distances, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_distances, ort_results[1], atol=1e-4)

    def test_birch_two_clusters(self):
        from sklearn.cluster import Birch
        from yobx.sklearn import to_onnx

        X = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        model = Birch(n_clusters=2)
        model.fit(X)

        onx = to_onnx(model, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = model.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

    def test_birch_com_microsoft_cdist(self):
        from sklearn.cluster import Birch
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        model = Birch(n_clusters=3)
        model.fit(X)

        onx = to_onnx(model, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected_labels = model.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_birch_labels_dtypes(self):
        from sklearn.cluster import Birch
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X32 = rng.standard_normal((30, 4)).astype(np.float32)
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                X = X32.astype(dtype)
                model = Birch(n_clusters=3)
                model.fit(X)
                onx = to_onnx(model, (X,))

                ref = ExtendedReferenceEvaluator(onx)
                results = ref.run(None, {"X": X})
                labels = results[0]

                expected_labels = model.predict(X).astype(np.int64)
                self.assertEqualArray(expected_labels, labels)

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                self.assertEqualArray(expected_labels, ort_results[0])

    def test_birch_distances_dtypes(self):
        from sklearn.cluster import Birch
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X32 = rng.standard_normal((20, 3)).astype(np.float32)
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                X = X32.astype(dtype)
                model = Birch(n_clusters=4)
                model.fit(X)
                onx = to_onnx(model, (X,))

                ref = ExtendedReferenceEvaluator(onx)
                results = ref.run(None, {"X": X})
                distances = results[1]

                centers = model.subcluster_centers_.astype(dtype)
                expected_distances = np.sqrt(
                    np.sum(
                        (X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2
                    )
                )
                self.assertEqualArray(expected_distances, distances, atol=1e-4)

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                self.assertEqualArray(expected_distances, ort_results[1], atol=1e-4)

    def test_birch_com_microsoft_cdist_dtypes(self):
        from sklearn.cluster import Birch
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X32 = rng.standard_normal((30, 4)).astype(np.float32)
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                X = X32.astype(dtype)
                model = Birch(n_clusters=3)
                model.fit(X)
                onx = to_onnx(model, (X,), target_opset={"": 18, "com.microsoft": 1})

                op_types = [(n.op_type, n.domain) for n in onx.graph.node]
                self.assertIn(("CDist", "com.microsoft"), op_types)

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                expected_labels = model.predict(X).astype(np.int64)
                self.assertEqualArray(expected_labels, ort_results[0])

    def test_birch_pipeline(self):
        from sklearn.cluster import Birch
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("birch", Birch(n_clusters=3)),
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
