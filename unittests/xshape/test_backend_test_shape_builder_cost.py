"""Checks arithmetic costs against native backend test models."""

import unittest
from onnx_light.onnx.backend import collect_test_cases

from yobx.xshape import InferenceMode, NativeShapeInference


class TestNativeBackendCosts(unittest.TestCase):
    def test_backend_costs(self):
        for op in ("Add", "MatMul", "Gemm", "Relu", "Sigmoid", "Softmax", "Conv"):
            for case in collect_test_cases(op):
                with self.subTest(case=case.name):
                    inference = NativeShapeInference()
                    costs = inference.run_model(case.model, inference=InferenceMode.COST)
                    self.assertEqual(len(costs), len(case.model.graph.node))
                    for name, flops, shapes in costs:
                        self.assertIsInstance(name, str)
                        self.assertIsInstance(shapes, tuple)
                        self.assertIsInstance(flops, int)
                        self.assertGreaterEqual(flops, 0)


if __name__ == "__main__":
    unittest.main()
