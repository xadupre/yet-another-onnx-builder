"""
Unit tests for the VotingClassifier and VotingRegressor converters.
"""

import unittest
import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnVotingClassifier(ExtTestCase):
    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [0, 1], [9, 10]],
        dtype=np.float32,
    )
    _y_bin = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    _y_multi = np.array([0, 0, 1, 1, 2, 2, 0, 1])

    def _check_soft(self, X, y, weights=None, atol=1e-5):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            clf = VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(random_state=0, max_iter=1000)),
                    ("dt", DecisionTreeClassifier(random_state=0)),
                ],
                voting="soft",
                weights=weights,
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

    def _check_hard(self, X, y, weights=None, atol=1e-5):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            clf = VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(random_state=0, max_iter=1000)),
                    ("dt", DecisionTreeClassifier(random_state=0)),
                    ("dt2", DecisionTreeClassifier(random_state=1)),
                ],
                voting="hard",
                weights=weights,
            )
            clf.fit(Xd, y)
            onx = to_onnx(clf, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label = results[0]

            expected_label = clf.predict(Xd)
            self.assertEqualArray(expected_label, label)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])

    def test_soft_binary(self):
        self._check_soft(self._X, self._y_bin)

    def test_soft_multiclass(self):
        self._check_soft(self._X, self._y_multi)

    def test_soft_weighted(self):
        self._check_soft(self._X, self._y_multi, weights=[2, 1])

    def test_hard_binary(self):
        self._check_hard(self._X, self._y_bin)

    def test_hard_multiclass(self):
        self._check_hard(self._X, self._y_multi)

    def test_hard_weighted(self):
        self._check_hard(self._X, self._y_multi, weights=[3, 1, 1])

    def test_soft_in_pipeline(self):
        """VotingClassifier (soft) as last step in a Pipeline."""
        Xd = self._X.astype(np.float32)
        y = self._y_multi
        clf = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(random_state=0, max_iter=1000)),
                ("dt", DecisionTreeClassifier(random_state=0)),
            ],
            voting="soft",
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


@requires_sklearn("1.4")
class TestSklearnVotingRegressor(ExtTestCase):
    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [0, 1], [9, 10]],
        dtype=np.float32,
    )
    _y = np.array([1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 1.0, 5.0], dtype=np.float32)

    def _check_reg(self, X, y, weights=None, atol=1e-5):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            yd = y.astype(dtype)
            reg = VotingRegressor(
                estimators=[
                    ("lr", LinearRegression()),
                    ("ridge", Ridge()),
                    ("dt", DecisionTreeRegressor(random_state=0)),
                ],
                weights=weights,
            )
            reg.fit(Xd, yd)
            onx = to_onnx(reg, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            pred = results[0].ravel()

            expected_pred = reg.predict(Xd).astype(dtype)
            self.assertEqualArray(expected_pred, pred, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_pred, ort_results[0].ravel(), atol=atol)

    def test_regressor_equal_weights(self):
        self._check_reg(self._X, self._y)

    def test_regressor_weighted(self):
        self._check_reg(self._X, self._y, weights=[2, 1, 1])

    def test_regressor_in_pipeline(self):
        """VotingRegressor as last step in a Pipeline."""
        Xd = self._X.astype(np.float32)
        reg = VotingRegressor(
            estimators=[("lr", LinearRegression()), ("ridge", Ridge())]
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
        pipe.fit(Xd, self._y)

        onx = to_onnx(pipe, (Xd,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": Xd})
        expected_pred = pipe.predict(Xd)
        self.assertEqualArray(expected_pred, results[0].ravel(), atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": Xd})
        self.assertEqualArray(expected_pred, ort_results[0].ravel(), atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
