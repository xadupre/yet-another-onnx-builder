"""Checks native inference against the published wheel's backend test models."""

import unittest
from onnx_light.onnx.backend import collect_test_cases

from yobx.xshape import NativeShapeInference


class TestNativeBackendShapes(unittest.TestCase):
    def test_backend_shapes(self):
        for op in (
            "Add",
            "MatMul",
            "Gemm",
            "Reshape",
            "Shape",
            "Transpose",
            "Squeeze",
            "Unsqueeze",
        ):
            for case in collect_test_cases(op):
                with self.subTest(case=case.name):
                    if case.name == "test_cc_squeeze_all_singleton":
                        self.skipTest(
                            "onnx-light 0.1.26 does not infer Squeeze with an empty "
                            "axes graph input; omitting the axes input is supported."
                        )
                    model = case.model
                    inference = NativeShapeInference()
                    inference.run_model(model)
                    for output in model.graph.output:
                        tensor = output.type.tensor_type
                        if tensor.HasField("shape"):
                            expected = tuple(
                                str(dim.dim_param) if dim.dim_param else dim.dim_value
                                for dim in tensor.shape.dim
                            )
                            self.assertEqual(inference.get_shape(str(output.name)), expected)


if __name__ == "__main__":
    unittest.main()
