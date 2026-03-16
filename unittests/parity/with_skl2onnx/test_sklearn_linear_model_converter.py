import unittest
import numpy as np
import onnx
import onnx.helper as oh
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from yobx import DEFAULT_TARGET_OPSET as TARGET_OPSET
from yobx.ext_test_case import ExtTestCase
from yobx.sklearn.tests_helper import dump_data_and_model
from yobx.sklearn import to_onnx
import onnxruntime


class TestSklearnLinearModelConverters(ExtTestCase):
    def test_model_linear_regression(self):
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X = X.astype(np.float64)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.DOUBLE, [None, X_train.shape[1]]
                )
            ],
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
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.DOUBLE, [None, X_train.shape[1]]
                )
            ],
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
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.DOUBLE, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnLogisticRegression")

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
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.DOUBLE, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx, basename="SklearnLogisticRegressionMulticlass"
        )


if __name__ == "__main__":
    unittest.main()
