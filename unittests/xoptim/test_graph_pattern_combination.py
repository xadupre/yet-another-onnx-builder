import unittest

import numpy as np
from onnx_light import onnx
from onnx_light.onnx import checker, helper, numpy_helper
from onnx_light.onnx.reference import ReferenceEvaluator

from yobx.ext_test_case import ExtTestCase
from yobx.xbuilder.graph_builder import GraphBuilder, OptimizationOptions


class TestGraphPatternCombination(ExtTestCase):
    def _range(self, *shape):
        size = np.prod(shape)
        return (np.arange(size, dtype=np.float32) / size).astype(np.float32).reshape(shape)

    def _make_model(self, dynamic=False, keep_intermediate=False):
        dim32 = "D32" if dynamic else 32
        dim128 = "D128" if dynamic else 128
        batch = "batch" if dynamic else 3
        channel = "channel" if dynamic else 5
        outputs = [
            helper.make_tensor_value_info(
                "Z", onnx.TensorProto.FLOAT, [batch, channel, dim32, 64]
            )
        ]
        if keep_intermediate:
            outputs.append(
                helper.make_tensor_value_info("xm1", onnx.TensorProto.FLOAT, [1, dim32, dim128])
            )
        return helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    helper.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    helper.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    helper.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    helper.make_node("Cast", ["xm2c"], ["xm2"], to=onnx.TensorProto.FLOAT),
                    helper.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    helper.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "reshape-matmul-reshape",
                [
                    helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [dim32, dim128]),
                    helper.make_tensor_value_info(
                        "Y", onnx.TensorProto.FLOAT, [batch, channel, dim128, 64]
                    ),
                ],
                outputs,
                [
                    numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero"),
                    numpy_helper.from_array(np.array([1], dtype=np.int64), name="un"),
                    numpy_helper.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    numpy_helper.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    numpy_helper.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )

    def _optimize(self, model):
        builder = GraphBuilder(
            model,
            optimization_options=OptimizationOptions(
                patterns=[
                    "Cast",
                    "ReshapeMatMulReshape",
                    "UnsqueezeUnsqueeze",
                    "MatMulReshape2Of3",
                    "ReshapeReshape",
                ]
            ),
        )
        return builder.to_onnx(optimize=True).proto

    def _check_equivalent(self, model, optimized):
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        expected = ReferenceEvaluator(model).run(None, feeds)
        actual = ReferenceEvaluator(optimized).run(None, feeds)
        self.assertEqual(len(expected), len(actual))
        for expected_value, actual_value in zip(expected, actual):
            self.assertEqualArray(expected_value, actual_value)

    def test_reshape_matmul_reshape_static(self):
        model = self._make_model()
        checker.check_model(model)
        optimized = self._optimize(model)
        self.assertEqual(["Unsqueeze", "MatMul"], [node.op_type for node in optimized.graph.node])
        self.assertEqual(1, len(optimized.graph.initializer))
        self._check_equivalent(model, optimized)

    def test_reshape_matmul_reshape_keeps_graph_output(self):
        model = self._make_model(keep_intermediate=True)
        checker.check_model(model)
        optimized = self._optimize(model)
        self.assertEqual(
            ["Unsqueeze", "Reshape", "MatMul"], [node.op_type for node in optimized.graph.node]
        )
        self.assertEqual(2, len(optimized.graph.initializer))
        self._check_equivalent(model, optimized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
