import unittest
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnAdaBoostClassifier(ExtTestCase):
    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [0, 1], [9, 10]], dtype=np.float32
    )
    _y_bin = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    _y_multi = np.array([0, 0, 1, 1, 2, 2, 0, 1])

    def _check_classifier(self, X, y, n_estimators=5, atol=1e-5):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=0)
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

    @requires_sklearn("1.8")
    def test_binary(self):
        self._check_classifier(self._X, self._y_bin)

    @requires_sklearn("1.8")
    def test_binary_more_estimators(self):
        self._check_classifier(self._X, self._y_bin, n_estimators=10)

    @requires_sklearn("1.8")
    def test_multiclass(self):
        self._check_classifier(self._X, self._y_multi, n_estimators=10)

    @requires_sklearn("1.8")
    def test_in_pipeline(self):
        rng = np.random.default_rng(42)
        y = self._y_bin
        for dtype in (np.float32, np.float64):
            Xd = rng.standard_normal((8, 2)).astype(dtype)
            clf = AdaBoostClassifier(n_estimators=5, random_state=0)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(Xd, y)

            onx = to_onnx(pipe, (Xd,))
            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected_label = pipe.predict(Xd)
            expected_proba = pipe.predict_proba(Xd).astype(dtype)
            self.assertEqualArray(expected_label, results[0])
            self.assertEqualArray(expected_proba, results[1], atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    @requires_sklearn("1.8")
    def test_custom_base_estimator(self):
        y = self._y_bin
        for dtype in (np.float32, np.float64):
            Xd = self._X.astype(dtype)
            clf = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2), n_estimators=5, random_state=0
            )
            clf.fit(Xd, y)
            onx = to_onnx(clf, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected_label = clf.predict(Xd)
            expected_proba = clf.predict_proba(Xd).astype(dtype)
            self.assertEqualArray(expected_label, results[0])
            self.assertEqualArray(expected_proba, results[1], atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)


@requires_sklearn("1.4")
class TestSklearnAdaBoostRegressor(ExtTestCase):
    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [0, 1], [9, 10]], dtype=np.float32
    )
    _y = np.array([1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 1.0, 5.0], dtype=np.float32)

    def _check_regressor(self, X, y, n_estimators=5, atol=1e-5):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            yd = y.astype(dtype)
            reg = AdaBoostRegressor(n_estimators=n_estimators, random_state=0)
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

    def test_regressor_default(self):
        self._check_regressor(self._X, self._y)

    def test_regressor_more_estimators(self):
        self._check_regressor(self._X, self._y, n_estimators=10)

    def test_in_pipeline(self):
        rng = np.random.default_rng(42)
        for dtype in (np.float32, np.float64):
            Xd = rng.standard_normal((8, 2)).astype(dtype)
            yd = self._y.astype(dtype)
            reg = AdaBoostRegressor(n_estimators=5, random_state=0)
            pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
            pipe.fit(Xd, yd)

            onx = to_onnx(pipe, (Xd,))
            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected_pred = pipe.predict(Xd).astype(dtype)
            self.assertEqualArray(expected_pred, results[0].ravel(), atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_pred, ort_results[0].ravel(), atol=1e-5)

    def test_custom_base_estimator(self):
        for dtype in (np.float32, np.float64):
            Xd = self._X.astype(dtype)
            yd = self._y.astype(dtype)
            reg = AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=2), n_estimators=5, random_state=0
            )
            reg.fit(Xd, yd)
            onx = to_onnx(reg, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected_pred = reg.predict(Xd).astype(dtype)
            self.assertEqualArray(expected_pred, results[0].ravel(), atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_pred, ort_results[0].ravel(), atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
