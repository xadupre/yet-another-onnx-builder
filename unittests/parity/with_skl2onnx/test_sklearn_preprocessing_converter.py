import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from yobx import DEFAULT_TARGET_OPSET as TARGET_OPSET
from yobx.ext_test_case import ExtTestCase
from yobx.sklearn.tests_helper import dump_data_and_model


class TestSklearnPreprocessingConverters(ExtTestCase):
    def _make_data(self):
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test = train_test_split(X, test_size=0.5, random_state=42)
        return X_train, X_test

    def test_model_standard_scaler(self):
        X_train, X_test = self._make_data()
        model = StandardScaler()
        model.fit(X_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn standard scaler",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnStandardScaler")

    def test_model_min_max_scaler(self):
        X_train, X_test = self._make_data()
        model = MinMaxScaler()
        model.fit(X_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn min max scaler",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnMinMaxScaler")

    def test_model_normalizer(self):
        X_train, X_test = self._make_data()
        model = Normalizer()
        model.fit(X_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnNormalizer")

    def test_model_max_abs_scaler(self):
        X_train, X_test = self._make_data()
        model = MaxAbsScaler()
        model.fit(X_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn max abs scaler",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnMaxAbsScaler")


if __name__ == "__main__":
    unittest.main()
