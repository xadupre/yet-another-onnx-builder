import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from yobx import DEFAULT_TARGET_OPSET as TARGET_OPSET
from yobx.sklearn.tests_helper import dump_data_and_model


class TestSklearnSVMConverters(unittest.TestCase):
    def test_model_linear_svc(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = LinearSVC(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn linear svc",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnLinearSVC")

    def test_model_linear_svc_multiclass(self):
        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = LinearSVC(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn linear svc multiclass",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx, basename="SklearnLinearSVCMulticlass"
        )


if __name__ == "__main__":
    unittest.main()
