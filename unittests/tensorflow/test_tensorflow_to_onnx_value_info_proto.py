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

    def test_native_captured_weights_and_constants(self):
        """Converts captured variables and constants without TensorFlow initializer objects."""
        from onnx_light import onnx
        from yobx.builder.onnxlight import OnnxLightGraphBuilder

        class NativeArrayBuilder(OnnxLightGraphBuilder):
            def make_initializer(self, name, value, **kwargs):
                if not isinstance(value, (np.ndarray, onnx.TensorProto)):
                    raise TypeError(f"Expected a native initializer value, got {type(value)}")
                return super().make_initializer(name, value, **kwargs)

        weights = tf.Variable([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

        @tf.function
        def model(x):
            return tf.matmul(x, weights) + tf.constant([0.5, -0.5])

        value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        spec = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 2])
        artifact = to_onnx(model, (spec,), builder_cls=NativeArrayBuilder)
        onnx.checker.check_model(artifact.proto)
        expected = model(value).numpy()
        feeds = {"X:0": value}
        self.assertEqualArray(expected, ExtendedReferenceEvaluator(artifact).run(None, feeds)[0])
        self.assertEqualArray(expected, self.check_ort(artifact).run(None, feeds)[0])

    def test_shape_and_stack_preserve_dtypes(self):
        """Preserves TensorFlow shape and numeric stack output dtypes."""
        from onnx_light import onnx

        value = np.array([[0.5, 1.25], [-0.75, 2.0]], dtype=np.float32)
        spec = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 2])
        for dtype in (tf.int32, tf.int64):
            with self.subTest(shape_dtype=dtype.name):

                @tf.function
                def model(x):
                    return tf.shape(x, out_type=dtype), tf.stack([x, x + 0.25], axis=1)

                artifact = to_onnx(model, (spec,))
                onnx.checker.check_model(artifact.proto)
                expected = [output.numpy() for output in model(value)]
                feeds = {"X:0": value}
                for outputs in (
                    ExtendedReferenceEvaluator(artifact).run(None, feeds),
                    self.check_ort(artifact).run(None, feeds),
                ):
                    for want, got in zip(expected, outputs):
                        self.assertEqual(want.dtype, got.dtype)
                        self.assertEqualArray(want, got)

    def test_dense_value_info_proto(self):
        """ValueInfoProto replaces the numpy array as input specification."""
        from yobx._onnx_shim import onnx

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
        from yobx._onnx_shim import onnx

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
