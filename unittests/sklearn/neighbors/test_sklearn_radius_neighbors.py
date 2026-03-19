"""
Unit tests for yobx.sklearn.neighbors RadiusNeighbors converters.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestRadiusNeighborsClassifier(ExtTestCase):
    def test_rnn_classifier_basic(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Greater", op_types)
        self.assertIn("Not", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_float64(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(10)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0)
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

    def test_rnn_classifier_probabilities(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0)
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

    def test_rnn_classifier_multiclass(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] * 2).astype(np.int64) % 3  # 3 classes

        clf = RadiusNeighborsClassifier(radius=2.0)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        labels = results[0]

        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

    def test_rnn_classifier_outlier_label(self):
        """Test that outlier_label is applied to points with no neighbours."""
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X_train[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=0.5, outlier_label="most_frequent")
        clf.fit(X_train, y)

        # Include points that are guaranteed to be outliers (very far from training data)
        X_test = np.vstack(
            [
                X_train[:5],
                np.array(
                    [[100.0, 100.0, 100.0, 100.0], [-100.0, -100.0, -100.0, -100.0]],
                    dtype=np.float32,
                ),
            ]
        )

        onx = to_onnx(clf, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        labels = results[0]

        expected_labels = clf.predict(X_test).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_outlier_label_explicit(self):
        """Test explicit numeric outlier_label."""
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X_train = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X_train[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=0.5, outlier_label=1)
        clf.fit(X_train, y)

        X_test = np.array([[100.0, 100.0, 100.0, 100.0]], dtype=np.float32)

        onx = to_onnx(clf, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_test})
        labels = results[0]

        expected_labels = clf.predict(X_test).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_pipeline(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", RadiusNeighborsClassifier(radius=2.0))]
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

    def test_rnn_classifier_com_microsoft(self):
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0)
        clf.fit(X, y)

        onx = to_onnx(clf, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.proto.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        domains = {oi.domain for oi in onx.proto.opset_import}
        self.assertIn("com.microsoft", domains)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_metrics(self):
        """Test various distance metrics."""
        from sklearn.neighbors import RadiusNeighborsClassifier
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
                clf = RadiusNeighborsClassifier(radius=2.0, metric=metric, **kwargs)
                clf.fit(X, y)

                onx = to_onnx(clf, (X,))

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                expected_labels = clf.predict(X).astype(np.int64)
                self.assertEqualArray(expected_labels, ort_results[0])

    def test_rnn_classifier_opset_too_low_raises(self):
        """Opset < 13 must raise NotImplementedError."""
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(7)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0)
        clf.fit(X, y)

        with self.assertRaises(NotImplementedError):
            to_onnx(clf, (X,), target_opset=12)

    def test_rnn_classifier_non_uniform_weights_raises(self):
        """weights != 'uniform' must raise NotImplementedError."""
        from sklearn.neighbors import RadiusNeighborsClassifier
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(8)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = RadiusNeighborsClassifier(radius=2.0, weights="distance")
        clf.fit(X, y)

        with self.assertRaises(NotImplementedError):
            to_onnx(clf, (X,))


@requires_sklearn("1.4")
class TestRadiusNeighborsRegressor(ExtTestCase):
    def test_rnn_regressor_basic(self):
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = X[:, 0] * 2 + 1

        reg = RadiusNeighborsRegressor(radius=2.0)
        reg.fit(X, y)

        onx = to_onnx(reg, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Greater", op_types)
        self.assertIn("Not", op_types)
        self.assertIn("Div", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected = reg.predict(X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_rnn_regressor_float64(self):
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(20)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        y = (X[:, 0] * 2 + 1).astype(np.float64)

        reg = RadiusNeighborsRegressor(radius=2.0)
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

    def test_rnn_regressor_pipeline(self):
        from sklearn.neighbors import RadiusNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = X[:, 0] + X[:, 1]

        pipe = Pipeline(
            [("scaler", StandardScaler()), ("reg", RadiusNeighborsRegressor(radius=2.0))]
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

    def test_rnn_regressor_com_microsoft(self):
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = X[:, 0] * 2 + 1

        reg = RadiusNeighborsRegressor(radius=2.0)
        reg.fit(X, y)

        onx = to_onnx(reg, (X,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.proto.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected = reg.predict(X).astype(np.float32)
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_rnn_regressor_metrics(self):
        """Test various distance metrics."""
        from sklearn.neighbors import RadiusNeighborsRegressor
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
                reg = RadiusNeighborsRegressor(radius=2.0, metric=metric, **kwargs)
                reg.fit(X, y)

                onx = to_onnx(reg, (X,))

                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": X})
                expected = reg.predict(X).astype(np.float32)
                self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_rnn_regressor_outlier_nan(self):
        """Points with no neighbours within radius produce NaN predictions."""
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((30, 4)).astype(np.float32)
        y = X_train[:, 0] * 2 + 1

        reg = RadiusNeighborsRegressor(radius=2.0)
        reg.fit(X_train, y)

        # A point far from all training data should produce NaN
        X_outlier = np.array([[100.0, 100.0, 100.0, 100.0]], dtype=np.float32)

        onx = to_onnx(reg, (X_train,))

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_outlier})
        self.assertTrue(np.isnan(ort_results[0][0]))

    def test_rnn_regressor_opset_too_low_raises(self):
        """Opset < 13 must raise NotImplementedError."""
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y = X[:, 0] + 1

        reg = RadiusNeighborsRegressor(radius=2.0)
        reg.fit(X, y)

        with self.assertRaises(NotImplementedError):
            to_onnx(reg, (X,), target_opset=12)

    def test_rnn_regressor_non_uniform_weights_raises(self):
        """weights != 'uniform' must raise NotImplementedError."""
        from sklearn.neighbors import RadiusNeighborsRegressor
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y = X[:, 0] + 1

        reg = RadiusNeighborsRegressor(radius=2.0, weights="distance")
        reg.fit(X, y)

        with self.assertRaises(NotImplementedError):
            to_onnx(reg, (X,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
