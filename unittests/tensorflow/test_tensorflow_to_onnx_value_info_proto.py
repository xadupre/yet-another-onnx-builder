"""
Unit tests for ValueInfoProto support in yobx.tensorflow.to_onnx.
"""

import unittest
import numpy as np
import tensorflow as tf
from yobx.ext_test_case import ExtTestCase, requires_tensorflow
from yobx.reference import ExtendedReferenceEvaluator
from yobx.tensorflow import to_onnx


@requires_tensorflow("2.18")
class TestTensorflowToOnnxValueInfoProto(ExtTestCase):
    """Tests that to_onnx accepts :class:`onnx.ValueInfoProto` as input descriptors."""

    def test_dense_value_info_proto(self):
        """ValueInfoProto replaces the numpy array as input specification."""
        import onnx

        model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
        X = np.random.rand(5, 3).astype(np.float32)

        vip = onnx.helper.make_tensor_value_info("my_input", onnx.TensorProto.FLOAT, [None, 3])
        onx = to_onnx(model, (vip,))

        # The graph input must carry the name from the ValueInfoProto.
        self.assertEqual(onx.graph.input[0].name, "my_input:0")

        feeds = {"my_input:0": X}
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, feeds)[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_dense_value_info_proto_with_input_names_override(self):
        """input_names overrides the name embedded in a ValueInfoProto."""
        import onnx

        model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
        X = np.random.rand(5, 3).astype(np.float32)

        vip = onnx.helper.make_tensor_value_info("my_input", onnx.TensorProto.FLOAT, [None, 3])
        onx = to_onnx(model, (vip,), input_names=["override_name"])

        self.assertEqual(onx.graph.input[0].name, "override_name:0")

        feeds = {"override_name:0": X}
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, feeds)[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
