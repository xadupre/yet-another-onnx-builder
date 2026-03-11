"""
Unit tests for the DummyRegressor and DummyClassifier converters.
"""

import unittest
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnDummyRegressor(ExtTestCase):

    _X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    _y = np.array([1.0, 2.0, 3.0, 4.0])

    def _check_regressor(self, estimator, X, atol=1e-5):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            estimator.fit(Xd, self._y)
            onx = to_onnx(estimator, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            pred_onnx = results[0]

            expected = estimator.predict(Xd).astype(dtype)
            self.assertEqualArray(expected, pred_onnx, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected, ort_results[0], atol=atol)

    def test_dummy_regressor_mean(self):
        self._check_regressor(DummyRegressor(strategy="mean"), self._X)

    def test_dummy_regressor_median(self):
        self._check_regressor(DummyRegressor(strategy="median"), self._X)

    def test_dummy_regressor_constant(self):
        self._check_regressor(DummyRegressor(strategy="constant", constant=99.0), self._X)

    def test_dummy_regressor_quantile(self):
        self._check_regressor(DummyRegressor(strategy="quantile", quantile=0.75), self._X)

    def test_dummy_regressor_multi_output(self):
        """DummyRegressor with multiple output targets."""
        X = self._X
        y_multi = np.column_stack([self._y, self._y * 10])
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            reg = DummyRegressor(strategy="mean")
            reg.fit(Xd, y_multi)
            onx = to_onnx(reg, (Xd,))

            expected = reg.predict(Xd).astype(dtype)
            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            self.assertEqualArray(expected, results[0], atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected, ort_results[0], atol=1e-5)


@requires_sklearn("1.4")
class TestSklearnDummyClassifier(ExtTestCase):

    _X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dtype=np.float32)
    _y = np.array([0, 0, 0, 1, 1, 2])  # class 0 most frequent

    def _check_classifier(self, estimator, X, y, atol=1e-5):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            estimator.fit(Xd, y)
            onx = to_onnx(estimator, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label_onnx, proba_onnx = results[0], results[1]

            expected_label = estimator.predict(Xd)
            expected_proba = estimator.predict_proba(Xd).astype(dtype)

            self.assertEqualArray(expected_label, label_onnx)
            self.assertEqualArray(expected_proba, proba_onnx, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_proba, ort_results[1], atol=atol)

    def test_dummy_classifier_most_frequent(self):
        self._check_classifier(
            DummyClassifier(strategy="most_frequent"), self._X, self._y
        )

    def test_dummy_classifier_prior(self):
        self._check_classifier(DummyClassifier(strategy="prior"), self._X, self._y)

    def test_dummy_classifier_constant(self):
        self._check_classifier(
            DummyClassifier(strategy="constant", constant=1), self._X, self._y
        )

    def test_dummy_classifier_binary(self):
        """Binary classification (two classes)."""
        y_bin = np.array([0, 0, 0, 1, 1, 1])
        self._check_classifier(
            DummyClassifier(strategy="most_frequent"), self._X, y_bin
        )

    def test_dummy_classifier_unsupported_strategy(self):
        """Unsupported strategies raise NotImplementedError."""
        for strategy in ("stratified", "uniform"):
            clf = DummyClassifier(strategy=strategy)
            clf.fit(self._X, self._y)
            with self.assertRaises(NotImplementedError):
                to_onnx(clf, (self._X,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
