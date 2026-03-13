"""
Unit tests for the GridSearchCV converter.
"""

import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnGridSearchCV(ExtTestCase):

    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [0, 1], [9, 10], [2, 3], [4, 5]],
        dtype=np.float32,
    )
    _y_bin = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    _y_reg = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 5.0, 1.5, 2.5])

    def _check_classifier(self, clf, X, dtypes=(np.float32, np.float64), atol=1e-5):
        """Convert, run, and compare classifier outputs."""
        for dtype in dtypes:
            Xd = X.astype(dtype)
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

    def _check_regressor(self, reg, X, dtypes=(np.float32, np.float64), atol=1e-5):
        """Convert, run, and compare regressor outputs."""
        for dtype in dtypes:
            Xd = X.astype(dtype)
            onx = to_onnx(reg, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            pred = results[0]

            expected_pred = reg.predict(Xd).reshape(-1, 1).astype(dtype)
            self.assertEqualArray(expected_pred, pred, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_pred, ort_results[0], atol=atol)

    def test_classifier_binary(self):
        clf = GridSearchCV(LogisticRegression(solver="lbfgs"), {"C": [0.1, 1.0, 10.0]}, cv=2).fit(
            self._X, self._y_bin
        )
        self._check_classifier(clf, self._X)

    def test_regressor(self):
        reg = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, cv=2).fit(self._X, self._y_reg)
        self._check_regressor(reg, self._X)

    def test_classifier_in_pipeline(self):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0]}, cv=2)),
            ]
        ).fit(self._X, self._y_bin)
        self._check_classifier(pipe, self._X)

    def test_classifier_svc(self):
        clf = GridSearchCV(SVC(kernel="linear", probability=True), {"C": [0.1, 1.0]}, cv=2).fit(
            self._X, self._y_bin
        )
        # SVC always outputs float32 probabilities; test only with float32 input.
        self._check_classifier(clf, self._X, dtypes=(np.float32,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
