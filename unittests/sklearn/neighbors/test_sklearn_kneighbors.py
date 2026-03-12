"""
Unit tests for yobx.sklearn.neighbors KNN converters.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestKNeighborsClassifier(ExtTestCase):
    def test_knn_classifier_basic(self):
        from sklearn.neighbors import KNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TopK", op_types)
        self.assertIn("Gather", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_knn_classifier_float64(self):
        from sklearn.neighbors import KNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(10)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_knn_classifier_more_k(self):
        """Test with larger k values."""
        from sklearn.neighbors import KNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(11)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        for k in [1, 5, 10, 15]:
            with self.subTest(k=k):
                clf = KNeighborsClassifier(n_neighbors=k)
                clf.fit(X, y)

                onx = to_onnx(clf, (X,))

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                expected_labels = clf.predict(X).astype(np.int64)
                self.assertEqualArray(expected_labels, ort_results[0])

    def test_knn_classifier_probabilities(self):
        from sklearn.neighbors import KNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels, probabilities = results[0], results[1]

        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        expected_proba = clf.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_proba, probabilities, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_knn_classifier_multiclass(self):
        from sklearn.neighbors import KNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] * 2).astype(np.int64) % 3  # 3 classes

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

    def test_knn_classifier_pipeline(self):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=3)),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = pipe.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_knn_classifier_com_microsoft(self):
        from sklearn.neighbors import KNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        # CDist from com.microsoft should appear in the graph
        self.assertIn(("CDist", "com.microsoft"), op_types)

        domains = {oi.domain for oi in onx.opset_import}
        self.assertIn("com.microsoft", domains)

        # Verify predictions match sklearn using ORT (which supports CDist)
        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_knn_classifier_metrics(self):
        """Test various distance metrics for the classifier."""
        from sklearn.neighbors import KNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(30)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        metrics = [
            ("euclidean", {}),
            ("cosine", {}),
            ("manhattan", {}),
            ("chebyshev", {}),
            ("minkowski", {"p": 3}),
        ]
        for metric, kwargs in metrics:
            with self.subTest(metric=metric, **kwargs):
                clf = KNeighborsClassifier(n_neighbors=5, metric=metric, **kwargs)
                clf.fit(X, y)

                onx = to_onnx(clf, (X,))

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                expected_labels = clf.predict(X).astype(np.int64)
                self.assertEqualArray(expected_labels, ort_results[0])

    def test_knn_classifier_metrics_minkowski_p1_p2(self):
        """minkowski with p=1/2 are normalised to manhattan/euclidean."""
        from sklearn.neighbors import KNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(31)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        for p in [1, 2]:
            with self.subTest(p=p):
                clf = KNeighborsClassifier(n_neighbors=3, metric="minkowski", p=p)
                clf.fit(X, y)

                onx = to_onnx(clf, (X,))
                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                expected = clf.predict(X).astype(np.int64)
                self.assertEqualArray(expected, ort_results[0])

    def test_knn_classifier_opset_too_low_raises(self):
        """Opset < 13 must raise NotImplementedError."""
        from sklearn.neighbors import KNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        with self.assertRaises(NotImplementedError):
            to_onnx(clf, (X,), target_opset=12)


@requires_sklearn("1.4")
class TestKNeighborsRegressor(ExtTestCase):
    def test_knn_regressor_basic(self):
        from sklearn.neighbors import KNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = X[:, 0] * 2 + 1

        reg = KNeighborsRegressor(n_neighbors=3)
        reg.fit(X, y)

        onx = to_onnx(reg, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TopK", op_types)
        self.assertIn("Gather", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = reg.predict(X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_knn_regressor_float64(self):
        from sklearn.neighbors import KNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(20)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        y = (X[:, 0] * 2 + 1).astype(np.float64)

        reg = KNeighborsRegressor(n_neighbors=3)
        reg.fit(X, y)

        onx = to_onnx(reg, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = reg.predict(X).astype(np.float64)
        self.assertEqualArray(expected, predictions, atol=1e-10)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-10)

    def test_knn_regressor_more_k(self):
        """Test with larger k values."""
        from sklearn.neighbors import KNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(21)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)

        for k in [1, 5, 10, 15]:
            with self.subTest(k=k):
                reg = KNeighborsRegressor(n_neighbors=k)
                reg.fit(X, y)

                onx = to_onnx(reg, (X,))

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                expected = reg.predict(X).astype(np.float32)
                self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_knn_regressor_pipeline(self):
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = X[:, 0] + X[:, 1]

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", KNeighborsRegressor(n_neighbors=5)),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = pipe.predict(X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_knn_regressor_com_microsoft(self):
        from sklearn.neighbors import KNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = X[:, 0] * 2 + 1

        reg = KNeighborsRegressor(n_neighbors=3)
        reg.fit(X, y)

        onx = to_onnx(reg, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        # CDist from com.microsoft should appear in the graph
        self.assertIn(("CDist", "com.microsoft"), op_types)

        domains = {oi.domain for oi in onx.opset_import}
        self.assertIn("com.microsoft", domains)

        # Verify predictions match sklearn using ORT
        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected = reg.predict(X).astype(np.float32)
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_knn_regressor_metrics(self):
        """Test various distance metrics for the regressor."""
        from sklearn.neighbors import KNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(40)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)

        metrics = [
            ("euclidean", {}),
            ("cosine", {}),
            ("manhattan", {}),
            ("chebyshev", {}),
            ("minkowski", {"p": 3}),
        ]
        for metric, kwargs in metrics:
            with self.subTest(metric=metric, **kwargs):
                reg = KNeighborsRegressor(n_neighbors=5, metric=metric, **kwargs)
                reg.fit(X, y)

                onx = to_onnx(reg, (X,))

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                expected = reg.predict(X).astype(np.float32)
                self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_knn_regressor_opset_too_low_raises(self):
        """Opset < 18 must raise NotImplementedError."""
        from sklearn.neighbors import KNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y = X[:, 0] + 1

        reg = KNeighborsRegressor(n_neighbors=3)
        reg.fit(X, y)

        with self.assertRaises(NotImplementedError):
            to_onnx(reg, (X,), target_opset=17)


if __name__ == "__main__":
    unittest.main(verbosity=2)
