import unittest
import numpy as np
import onnxruntime
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from yobx import DEFAULT_TARGET_OPSET as TARGET_OPSET
from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator


class TestSklearnEnsembleConverters(ExtTestCase):
    def test_model_random_forest_classifier(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn random forest classifier",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )
        self.assertTrue(model_onnx is not None)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)
        expected_labels = model.predict(X_test)
        expected_proba = model.predict_proba(X_test).astype(np.float32)
        np.testing.assert_array_equal(ort_out[0], expected_labels)
        np.testing.assert_allclose(ort_out[1], expected_proba, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(ref_out[0], expected_labels)
        np.testing.assert_allclose(ref_out[1], expected_proba, rtol=1e-5, atol=1e-5)

    def test_model_random_forest_regressor(self):
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn random forest regressor",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)[0].flatten()
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)[0].flatten()
        expected = model.predict(X_test).astype(np.float32)
        np.testing.assert_allclose(ort_out, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ref_out, expected, rtol=1e-5, atol=1e-5)

    def test_model_gradient_boosting_classifier(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn gradient boosting classifier",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )
        self.assertTrue(model_onnx is not None)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)
        expected_labels = model.predict(X_test)
        expected_proba = model.predict_proba(X_test).astype(np.float32)
        np.testing.assert_array_equal(ort_out[0], expected_labels)
        np.testing.assert_allclose(ort_out[1], expected_proba, rtol=1e-5, atol=1e-4)
        np.testing.assert_array_equal(ref_out[0], expected_labels)
        np.testing.assert_allclose(ref_out[1], expected_proba, rtol=1e-5, atol=1e-4)

    def test_model_gradient_boosting_regressor(self):
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn gradient boosting regressor",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)[0].flatten()
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)[0].flatten()
        expected = model.predict(X_test).astype(np.float32)
        np.testing.assert_allclose(ort_out, expected, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(ref_out, expected, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
