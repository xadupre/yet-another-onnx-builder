"""
Unit tests for yobx.sklearn.ensemble stacking converters.
"""

import unittest
import numpy as np
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


def _make_reg_data():
    rng = np.random.RandomState(0)
    X = rng.randn(30, 4).astype(np.float32)
    y = rng.randn(30).astype(np.float32)
    return X, y


def _make_binary_data():
    rng = np.random.RandomState(0)
    X = rng.randn(30, 4).astype(np.float32)
    y = (rng.randn(30) > 0).astype(int)
    return X, y


def _make_multiclass_data():
    rng = np.random.RandomState(0)
    X = rng.randn(30, 4).astype(np.float32)
    y = np.array([0, 1, 2] * 10)
    return X, y


@requires_sklearn("1.4")
class TestSklearnStackingRegressor(ExtTestCase):
    def test_stacking_regressor_basic(self):
        """StackingRegressor with two base regressors."""
        X, y = _make_reg_data()
        est = StackingRegressor(
            estimators=[
                ("dt", DecisionTreeRegressor(max_depth=2, random_state=0)),
                ("ridge", Ridge()),
            ],
            final_estimator=Ridge(),
        )
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        pred = results[0]

        expected = est.predict(X).astype(np.float32)
        self.assertEqualArray(expected, pred.ravel(), atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0].ravel(), atol=1e-4)

    def test_stacking_regressor_passthrough(self):
        """StackingRegressor with passthrough=True."""
        X, y = _make_reg_data()
        est = StackingRegressor(
            estimators=[
                ("dt", DecisionTreeRegressor(max_depth=2, random_state=0)),
                ("ridge", Ridge()),
            ],
            final_estimator=Ridge(),
            passthrough=True,
        )
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        pred = results[0]

        expected = est.predict(X).astype(np.float32)
        self.assertEqualArray(expected, pred.ravel(), atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0].ravel(), atol=1e-4)

    def test_stacking_regressor_pipeline_base(self):
        """StackingRegressor with a Pipeline as one of the base estimators."""
        X, y = _make_reg_data()
        est = StackingRegressor(
            estimators=[
                (
                    "pipe",
                    Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
                ),
                ("dt", DecisionTreeRegressor(max_depth=2, random_state=0)),
            ],
            final_estimator=Ridge(),
        )
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        pred = results[0]

        expected = est.predict(X).astype(np.float32)
        self.assertEqualArray(expected, pred.ravel(), atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0].ravel(), atol=1e-4)


@requires_sklearn("1.4")
class TestSklearnStackingClassifier(ExtTestCase):
    def test_stacking_classifier_binary(self):
        """StackingClassifier binary classification with predict_proba."""
        X, y = _make_binary_data()
        est = StackingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=2, random_state=0)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            final_estimator=LogisticRegression(max_iter=200),
        )
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = est.predict(X)
        expected_proba = est.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_stacking_classifier_multiclass(self):
        """StackingClassifier multiclass classification with predict_proba."""
        X, y = _make_multiclass_data()
        est = StackingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=2, random_state=0)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            final_estimator=LogisticRegression(max_iter=200),
        )
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = est.predict(X)
        expected_proba = est.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_stacking_classifier_binary_passthrough(self):
        """StackingClassifier binary with passthrough=True."""
        X, y = _make_binary_data()
        est = StackingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=2, random_state=0)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            final_estimator=LogisticRegression(max_iter=200),
            passthrough=True,
        )
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = est.predict(X)
        expected_proba = est.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_stacking_classifier_multiclass_passthrough(self):
        """StackingClassifier multiclass with passthrough=True."""
        X, y = _make_multiclass_data()
        est = StackingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=2, random_state=0)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            final_estimator=LogisticRegression(max_iter=200),
            passthrough=True,
        )
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = est.predict(X)
        expected_proba = est.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_stacking_classifier_pipeline_base(self):
        """StackingClassifier with a Pipeline as one of the base estimators."""
        X, y = _make_binary_data()
        est = StackingClassifier(
            estimators=[
                (
                    "pipe",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=200)),
                        ]
                    ),
                ),
                ("dt", DecisionTreeClassifier(max_depth=2, random_state=0)),
            ],
            final_estimator=LogisticRegression(max_iter=200),
        )
        est.fit(X, y)

        onx = to_onnx(est, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = est.predict(X)
        expected_proba = est.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
