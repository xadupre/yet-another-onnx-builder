"""
Unit tests for the TunedThresholdClassifierCV converter.
"""

import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.5")
class TestSklearnTunedThresholdClassifierCV(ExtTestCase):

    _X, _y = make_classification(n_samples=200, n_features=4, random_state=0)
    _X = _X.astype(np.float32)

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

    def test_prefit_logistic_regression(self):
        """TunedThresholdClassifierCV with cv='prefit' wrapping LogisticRegression."""
        base = LogisticRegression(solver="lbfgs").fit(self._X, self._y)
        clf = TunedThresholdClassifierCV(base, cv="prefit", refit=False)
        clf.fit(self._X, self._y)
        self._check_classifier(clf, self._X)

    def test_cv_logistic_regression(self):
        """TunedThresholdClassifierCV with default CV wrapping LogisticRegression."""
        clf = TunedThresholdClassifierCV(LogisticRegression(solver="lbfgs"))
        clf.fit(self._X, self._y)
        self._check_classifier(clf, self._X)

    def test_in_pipeline(self):
        """TunedThresholdClassifierCV used inside a Pipeline."""
        # Fit the inner estimator first (needed for cv='prefit').
        base_pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(solver="lbfgs"))]
        ).fit(self._X, self._y)
        clf = Pipeline(
            [
                ("scaler", base_pipe.named_steps["scaler"]),
                (
                    "clf",
                    TunedThresholdClassifierCV(
                        base_pipe.named_steps["clf"], cv="prefit", refit=False
                    ),
                ),
            ]
        )
        clf.fit(self._X, self._y)
        self._check_classifier(clf, self._X)

    def test_float64_input(self):
        """Converter works with float64 inputs."""
        X64 = self._X.astype(np.float64)
        base = LogisticRegression(solver="lbfgs").fit(X64, self._y)
        clf = TunedThresholdClassifierCV(base, cv="prefit", refit=False)
        clf.fit(X64, self._y)
        self._check_classifier(clf, X64, dtypes=(np.float64,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
