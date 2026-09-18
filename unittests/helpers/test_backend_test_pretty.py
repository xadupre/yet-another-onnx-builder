"""Exercises graph rendering with the backend cases shipped in the native wheel."""

import unittest
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.backend import collect_test_case
from yobx.helpers._onnx_simple_text_plot import _get_shape, _get_type
from yobx.helpers.onnx_helper import pretty_onnx


class TestPrettyNativeBackend(unittest.TestCase):
    def test_native_type_and_dimension_presence(self):
        """Preserves zero dimensions, unknown dimensions, and scalar rank."""
        for shape in ([], [0, "batch", None], None):
            with self.subTest(shape=shape):
                value = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
                self.assertEqual(_get_shape(value), shape)
        sequence = helper.make_sequence_type_proto(
            helper.make_tensor_type_proto(TensorProto.FLOAT, [2])
        )
        self.assertEqual(_get_type(sequence), "sequence(float32)")
        self.assertIsNone(_get_shape(sequence))

    def test_native_backend_models(self):
        """Renders native backend models without the reference ONNX test harness."""
        cases = collect_test_case(include_big=False)
        self.assertTrue(cases)
        for name, case in cases.items():
            with self.subTest(name=name):
                try:
                    self.assertIsNotNone(case.model)
                    self.assertTrue(pretty_onnx(case.model))
                finally:
                    case.unload()


if __name__ == "__main__":
    unittest.main(verbosity=2)
