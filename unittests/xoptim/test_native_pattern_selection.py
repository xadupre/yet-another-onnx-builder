import unittest

import numpy
from onnx_light import onnx
from onnx_light.onnx import checker
from onnx_light.onnx.reference import ReferenceEvaluator


class TestNativePatternSelection(unittest.TestCase):
    def make_builder(self, *args, **kwargs):
        from yobx.builder.onnxlight import OnnxLightGraphBuilder

        return OnnxLightGraphBuilder(*args, **kwargs)

    def test_native_pattern_names(self):
        from yobx.builder.onnxlight import OnnxLightOptimizationOptions

        default_names = OnnxLightOptimizationOptions().pattern_names()
        self.assertIn("TransposeTranspose", default_names)
        self.assertEqual(
            OnnxLightOptimizationOptions(patterns="default").pattern_names(), default_names
        )
        self.assertEqual(
            OnnxLightOptimizationOptions(patterns=["TransposeTranspose"]).pattern_names(),
            ["TransposeTranspose"],
        )
        self.assertEqual(OnnxLightOptimizationOptions(patterns=[]).pattern_names(), [])

    def test_default_explicit_and_empty_native_selection(self):
        from yobx.builder.onnxlight import OnnxLightOptimizationOptions

        cases = [
            ("default", None, 1, ["Identity"]),
            ("explicit", OnnxLightOptimizationOptions(["TransposeTranspose"]), 1, ["Identity"]),
            ("empty", OnnxLightOptimizationOptions([]), 0, ["Transpose", "Transpose"]),
        ]
        sample = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
        for label, optimization_options, expected_rewrites, expected_ops in cases:
            with self.subTest(label=label):
                builder = self.make_builder(18, optimization_options=optimization_options)
                builder.make_tensor_input("X", onnx.TensorProto.FLOAT, [2, 3])
                transposed = builder.op.Transpose("X", perm=[1, 0])
                output = builder.op.Transpose(transposed, perm=[1, 0], outputs="Y")
                builder.make_tensor_output(output)
                artifact = builder.to_onnx(return_optimize_report=True)
                checker.check_model(artifact.proto)
                self.assertEqual(artifact.report.extra["rewrites"], expected_rewrites)
                self.assertEqual(
                    [node.op_type for node in artifact.proto.graph.node], expected_ops
                )
                actual = ReferenceEvaluator(artifact.proto).run(None, {"X": sample})[0]
                numpy.testing.assert_array_equal(actual, sample)


if __name__ == "__main__":
    unittest.main()
