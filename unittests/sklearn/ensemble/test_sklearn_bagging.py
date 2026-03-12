"""
Unit tests for the BaggingClassifier and BaggingRegressor converters.
"""

import unittest
import numpy as np
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnBaggingRegressor(ExtTestCase):
    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [0, 1], [9, 10]],
        dtype=np.float32,
    )
    _y = np.array([1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 1.0, 5.0], dtype=np.float32)

    def _check_reg(self, X, y, **kwargs):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            yd = y.astype(dtype)
            reg = BaggingRegressor(
                estimator=DecisionTreeRegressor(random_state=0),
                random_state=0,
                **kwargs,
            )
            reg.fit(Xd, yd)
            onx = to_onnx(reg, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            pred = results[0].ravel()

            expected_pred = reg.predict(Xd).astype(dtype)
            self.assertEqualArray(expected_pred, pred, atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_pred, ort_results[0].ravel(), atol=1e-5)

    def test_regressor_default(self):
        """BaggingRegressor with default parameters."""
        self._check_reg(self._X, self._y)

    def test_regressor_max_features(self):
        """BaggingRegressor with max_features < n_features (feature subsampling)."""
        self._check_reg(self._X, self._y, max_features=1)

    def test_regressor_linear_base(self):
        """BaggingRegressor with LinearRegression as base estimator."""
        for dtype in (np.float32, np.float64):
            Xd = self._X.astype(dtype)
            yd = self._y.astype(dtype)
            reg = BaggingRegressor(
                estimator=LinearRegression(),
                n_estimators=5,
                random_state=0,
            )
            reg.fit(Xd, yd)
            onx = to_onnx(reg, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected_pred = reg.predict(Xd).astype(dtype)
            self.assertEqualArray(expected_pred, results[0].ravel(), atol=1e-5)

    def test_regressor_in_pipeline(self):
        """BaggingRegressor as last step in a Pipeline."""
        Xd = self._X.astype(np.float32)
        reg = BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=0),
            n_estimators=5,
            random_state=0,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
        pipe.fit(Xd, self._y)

        onx = to_onnx(pipe, (Xd,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": Xd})
        expected_pred = pipe.predict(Xd).astype(np.float32)
        self.assertEqualArray(expected_pred, results[0].ravel(), atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": Xd})
        self.assertEqualArray(expected_pred, ort_results[0].ravel(), atol=1e-5)


@requires_sklearn("1.4")
class TestSklearnBaggingClassifier(ExtTestCase):
    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [0, 1], [9, 10]],
        dtype=np.float32,
    )
    _y_bin = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    _y_multi = np.array([0, 0, 1, 1, 2, 2, 0, 1])

    def _check_clf(self, X, y, atol=1e-5, **kwargs):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            clf = BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=0),
                random_state=0,
                **kwargs,
            )
            clf.fit(Xd, y)
            onx = to_onnx(clf, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label, proba = results[0], results[1]

            expected_label = clf.predict(Xd)
            expected_proba = clf.predict_proba(Xd).astype(dtype)

            self.assertEqualArray(expected_label, label)
            self.assertEqualArray(expected_proba, proba, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_proba, ort_results[1], atol=atol)

    def test_classifier_binary(self):
        """BaggingClassifier binary classification."""
        self._check_clf(self._X, self._y_bin)

    def test_classifier_multiclass(self):
        """BaggingClassifier multiclass classification."""
        self._check_clf(self._X, self._y_multi)

    def test_classifier_max_features(self):
        """BaggingClassifier with max_features < n_features (feature subsampling)."""
        self._check_clf(self._X, self._y_multi, max_features=1)

    def test_classifier_logistic_base(self):
        """BaggingClassifier with LogisticRegression as base estimator."""
        for dtype in (np.float32, np.float64):
            Xd = self._X.astype(dtype)
            clf = BaggingClassifier(
                estimator=LogisticRegression(random_state=0, max_iter=1000),
                n_estimators=5,
                random_state=0,
            )
            clf.fit(Xd, self._y_bin)
            onx = to_onnx(clf, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected_label = clf.predict(Xd)
            self.assertEqualArray(expected_label, results[0])

    def test_classifier_in_pipeline(self):
        """BaggingClassifier as last step in a Pipeline."""
        Xd = self._X.astype(np.float32)
        y = self._y_multi
        clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=0),
            n_estimators=5,
            random_state=0,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(Xd, y)

        onx = to_onnx(pipe, (Xd,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": Xd})
        expected_label = pipe.predict(Xd)
        self.assertEqualArray(expected_label, results[0])

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": Xd})
        self.assertEqualArray(expected_label, ort_results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
