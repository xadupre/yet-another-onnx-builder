import unittest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType
from yobx import DEFAULT_TARGET_OPSET as TARGET_OPSET
from yobx.sklearn.tests_helper import dump_data_and_model
import onnxruntime


class TestSklearnLinearModelConverters(unittest.TestCase):
    def test_model_linear_regression(self):
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X = X.astype(np.float64)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn linear regression",
            [("input", DoubleTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_pred = sess.run(None, feeds)[0].flatten()
        expected = model.predict(X_test)
        np.testing.assert_allclose(ort_pred, expected, rtol=1e-5, atol=1e-5)

    def test_model_ridge_regression(self):
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X = X.astype(np.float64)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = Ridge(alpha=0.5)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn ridge regression",
            [("input", DoubleTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_pred = sess.run(None, feeds)[0].flatten()
        expected = model.predict(X_test)
        np.testing.assert_allclose(ort_pred, expected, rtol=1e-5, atol=1e-5)

    def test_model_logistic_regression(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float64)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn logistic regression",
            [("input", DoubleTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx, basename="SklearnLogisticRegression"
        )

    def test_model_logistic_regression_multiclass(self):
        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        X = X.astype(np.float64)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn logistic regression multiclass",
            [("input", DoubleTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx, basename="SklearnLogisticRegressionMulticlass"
        )


if __name__ == "__main__":
    unittest.main()
