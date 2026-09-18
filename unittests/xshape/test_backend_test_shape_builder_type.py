"""Checks native type inference without the reference ONNX backend harness."""

import unittest
from onnx_light.onnx.backend import collect_test_cases

from yobx.xshape import InferenceMode, NativeShapeInference


class TestNativeBackendTypes(unittest.TestCase):
    def test_backend_types(self):
        for op in ("Add", "Cast", "Equal", "MatMul", "Shape", "Reshape", "Transpose"):
            for case in collect_test_cases(op):
                with self.subTest(case=case.name):
                    inference = NativeShapeInference()
                    inference.run_model(case.model, inference=InferenceMode.TYPE)
                    for output in case.model.graph.output:
                        self.assertEqual(
                            inference.get_type(str(output.name)),
                            output.type.tensor_type.elem_type,
                        )
                        self.assertFalse(inference.has_shape(str(output.name)))


if __name__ == "__main__":
    unittest.main()
