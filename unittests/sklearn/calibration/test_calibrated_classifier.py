"""
Unit tests for the CalibratedClassifierCV converter.
"""

import unittest

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnCalibratedClassifierCV(ExtTestCase):

    _X_bin = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [0, 1], [9, 10], [2, 3], [4, 5]], dtype=np.float32
    )
    _y_bin = np.array([0, 0, 1, 1, 0, 1, 0, 1])

    _X_multi = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [0, 1], [6, 7], [1, 3]], dtype=np.float32
    )
    _y_multi = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    def _check(self, clf, X, atol=1e-5):
        """Convert, run, and compare outputs for both float32 and float64."""
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            onx = to_onnx(clf, (Xd,))

            output_names = [o.name for o in onx.proto.graph.output]
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

    def test_binary_sigmoid(self):
        clf = CalibratedClassifierCV(LogisticRegression(), cv=2, method="sigmoid").fit(
            self._X_bin, self._y_bin
        )
        self._check(clf, self._X_bin)

    def test_binary_isotonic(self):
        clf = CalibratedClassifierCV(LogisticRegression(), cv=2, method="isotonic").fit(
            self._X_bin, self._y_bin
        )
        self._check(clf, self._X_bin, atol=1e-4)

    def test_multiclass_sigmoid(self):
        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=500), cv=3, method="sigmoid"
        ).fit(self._X_multi, self._y_multi)
        self._check(clf, self._X_multi)

    def test_multiclass_isotonic(self):
        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=500), cv=3, method="isotonic"
        ).fit(self._X_multi, self._y_multi)
        self._check(clf, self._X_multi, atol=1e-4)

    def test_calibrated_in_pipeline(self):
        """CalibratedClassifierCV as the final step of a Pipeline."""
        X, y = self._X_bin, self._y_bin
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(LogisticRegression(), cv=2)),
            ]
        ).fit(X, y)
        onx = to_onnx(pipe, (X,))
        sess = self.check_ort(onx)
        ort_labels = sess.run(None, {"X": X})[0]
        self.assertEqualArray(pipe.predict(X), ort_labels)

    def test_sigmoid_with_pipeline_base_estimator_ensemble_false(self):
        X, y = make_classification(n_samples=600, n_features=20, random_state=42)
        X = np.abs(X).astype(np.float32)
        X_train, X_test = X[:450], X[450:]
        y_train = y[:450]

        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        base = Pipeline([("id", FunctionTransformer()), ("clf", gb)])
        clf = CalibratedClassifierCV(base, method="sigmoid", ensemble=False).fit(X_train, y_train)

        expected = clf.predict_proba(X_test).astype(X_test.dtype)
        onx = to_onnx(clf, (X_test,))
        got = self.check_ort(onx).run(None, {"X": X_test})[1]

        self.assertEqualArray(expected, got, atol=1e-5)

    def test_sigmoid_with_pipeline_base_estimator_ensemble_true(self):
        X, y = make_classification(n_samples=600, n_features=20, random_state=42)
        X = np.abs(X).astype(np.float32)
        X_train, X_test = X[:450], X[450:]
        y_train = y[:450]

        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        base = Pipeline([("id", FunctionTransformer()), ("clf", gb)])
        clf = CalibratedClassifierCV(base, method="sigmoid", ensemble=True).fit(X_train, y_train)

        expected = clf.predict_proba(X_test).astype(X_test.dtype)
        onx = to_onnx(clf, (X_test,))
        got = self.check_ort(onx).run(None, {"X": X_test})[1]

        self.assertEqualArray(expected, got, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
