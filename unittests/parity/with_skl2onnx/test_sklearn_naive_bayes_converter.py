import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from yobx import DEFAULT_TARGET_OPSET as TARGET_OPSET
from yobx.sklearn.tests_helper import dump_data_and_model


class TestSklearnNaiveBayesConverters(unittest.TestCase):
    def test_model_gaussian_nb(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn gaussian nb",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnGaussianNB")

    def test_model_gaussian_nb_multiclass(self):
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
        model = GaussianNB()
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn gaussian nb multiclass",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx, basename="SklearnGaussianNBMulticlass"
        )


if __name__ == "__main__":
    unittest.main()
