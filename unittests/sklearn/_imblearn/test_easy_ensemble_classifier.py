"""
Unit tests for the EasyEnsembleClassifier converter.
"""

import unittest
import numpy as np
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnEasyEnsembleClassifier(ExtTestCase):
    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [0, 1], [9, 10]], dtype=np.float32
    )
    _y_bin = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    _y_multi = np.array([0, 0, 1, 1, 2, 2, 0, 1])

    def _check_clf(self, X, y, atol=1e-5, **kwargs):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            clf = EasyEnsembleClassifier(random_state=0, **kwargs)
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
        """EasyEnsembleClassifier binary classification."""
        self._check_clf(self._X, self._y_bin)

    def test_classifier_multiclass(self):
        """EasyEnsembleClassifier multiclass classification."""
        self._check_clf(self._X, self._y_multi, n_estimators=2)

    def test_classifier_fewer_estimators(self):
        """EasyEnsembleClassifier with a smaller number of sub-estimators."""
        self._check_clf(self._X, self._y_bin, n_estimators=3)

    def test_classifier_decision_tree_base(self):
        """EasyEnsembleClassifier with a DecisionTreeClassifier base estimator."""
        from imblearn.ensemble import EasyEnsembleClassifier as EEC
        from sklearn.ensemble import AdaBoostClassifier

        for dtype in (np.float32, np.float64):
            Xd = self._X.astype(dtype)
            clf = EEC(
                estimator=AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1, random_state=0), random_state=0
                ),
                n_estimators=3,
                random_state=0,
            )
            clf.fit(Xd, self._y_bin)
            onx = to_onnx(clf, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected_label = clf.predict(Xd)
            self.assertEqualArray(expected_label, results[0])

    def test_classifier_in_pipeline(self):
        """EasyEnsembleClassifier as last step in a sklearn Pipeline."""
        Xd = self._X.astype(np.float32)
        y = self._y_bin
        clf = EasyEnsembleClassifier(n_estimators=3, random_state=0)
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
