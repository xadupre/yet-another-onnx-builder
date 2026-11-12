"""Exercises native backend cases through the explicit ONNX Runtime evaluator."""

import unittest
from unittest.mock import patch
import numpy
from onnx_light.onnx import helper
from onnx_light.onnx.backend import make_test_class
from yobx.reference.onnxruntime_evaluator import OnnxruntimeEvaluator


def run_model(model, *inputs):
    """Executes a native model through the evaluator's serialization boundary."""
    session = OnnxruntimeEvaluator(model, providers=["CPUExecutionProvider"])
    return session.run(None, dict(zip(session.input_names, inputs)))


TestGeneratedOnnxruntimeBackend = make_test_class(
    run_model,
    include_regex=[
        r"^test_add(?:_bcast)?$",
        r"^test_cc_matmul(?:_|$)",
        r"^test_cc_(scan|loop)_basic_trip_count$",
        r"^test_cc_loop_zero_trip_count$",
    ],
    include_big=False,
    unload=True,
)


class TestOnnxruntimeConstant(unittest.TestCase):
    def test_runtime_compatible_defaults(self):
        node = helper.make_node("Identity", ["X"], ["Y"])
        evaluator = OnnxruntimeEvaluator(node, providers=["CPUExecutionProvider"])
        values = numpy.array([1.0, -2.0], dtype=numpy.float32)
        model = evaluator._make_model_proto(
            [node],
            [helper.make_tensor_value_info("X", 1, [2])],
            [helper.make_tensor_value_info("Y", 1, [2])],
        )
        self.assertEqual(model.ir_version, 13)
        self.assertEqual({o.domain: o.version for o in model.opset_import}, {"": 26})
        numpy.testing.assert_array_equal(evaluator.run(None, {"X": values})[0], values)

    def test_explicit_runtime_versions(self):
        node = helper.make_node("Identity", ["X"], ["Y"])
        evaluator = OnnxruntimeEvaluator(node, ir_version=8, opsets=18)
        values = numpy.array([1.0, -2.0], dtype=numpy.float32)
        model = evaluator._make_model_proto(
            [node],
            [helper.make_tensor_value_info("X", 1, [2])],
            [helper.make_tensor_value_info("Y", 1, [2])],
        )
        self.assertEqual(model.ir_version, 8)
        self.assertEqual({o.domain: o.version for o in model.opset_import}, {"": 18})
        numpy.testing.assert_array_equal(evaluator.run(None, {"X": values})[0], values)

    def test_constant_tensor_boundary(self):
        node = helper.make_node("Constant", [], ["value"], value_float=1.0)
        actual = OnnxruntimeEvaluator(node, opsets=18).run(None, {})[0]
        numpy.testing.assert_array_equal(actual, numpy.array(1.0, dtype=numpy.float32))
        with patch(
            "yobx.reference.onnxruntime_evaluator.ExtendedReferenceEvaluator"
        ) as evaluator:
            evaluator.return_value.run.return_value = [[actual]]
            with self.assertRaisesRegex(TypeError, "must produce a tensor"):
                OnnxruntimeEvaluator(node, opsets=18).run(None, {})
