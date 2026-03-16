import unittest
import numpy as np
from sklearn.preprocessing import Binarizer
import onnx
import onnx.helper as oh
from yobx import DEFAULT_TARGET_OPSET as TARGET_OPSET
from yobx.ext_test_case import ExtTestCase
from yobx.sklearn import to_onnx
from yobx.sklearn.tests_helper import dump_data_and_model


class TestSklearnBinarizer(ExtTestCase):
    def test_model_binarizer(self):
        data = np.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]], dtype=np.float32)
        model = Binarizer(threshold=0.5)
        model.fit(data)
        model_onnx = to_onnx(
            model,
            [oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None, data.shape[1]])],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnBinarizer-SkipDim1")


if __name__ == "__main__":
    unittest.main()
