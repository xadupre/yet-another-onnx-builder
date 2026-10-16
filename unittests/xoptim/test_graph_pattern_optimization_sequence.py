import os
import unittest

import numpy
import onnx_light
import pytest
from packaging.version import Version
from onnx_light import onnx
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light.onnx_core.optimization import GraphGraph
from onnx_light.onnx.reference import ReferenceEvaluator

from yobx.ext_test_case import ExtTestCase


class TestGraphPatternOptimizationSequence(ExtTestCase):
    @pytest.mark.xfail(
        Version(onnx_light.__version__) < Version("0.1.33"),
        reason="xadupre/onnx-light#5299: Split sizes do not cover the input dimension",
        raises=ValueError,
        strict=True,
    )
    def test_sequences_split(self):
        data = os.path.join(os.path.dirname(__file__), "data", "sequences.onnx")
        model = onnx.load(data)
        values = numpy.arange(8400 * 80, dtype=numpy.float32).reshape(1, 8400, 80)
        feeds = {"permute_1": values}
        expected = [values[:, :5000], values[:, 5000:]]
        original_outputs = ReferenceEvaluator(model).run(None, feeds)
        self.assertEqual(len(expected), len(original_outputs))
        for want, got in zip(expected, original_outputs):
            self.assertEqualArray(want, got)
        builder = GraphBuilder(model)
        optimizer = GraphGraph(builder, patterns=["SplitToSequenceSequenceAt"])
        optimizer.optimize()
        optimized = builder.to_model(model.ir_version)
        op_types = [node.op_type for node in optimized.graph.node]
        self.assertIn("Split", op_types)
        self.assertNotIn("SplitToSequence", op_types)
        self.assertNotIn("SequenceAt", op_types)
        optimized_outputs = ReferenceEvaluator(optimized).run(None, feeds)
        self.assertEqual(len(expected), len(optimized_outputs))
        for want, got in zip(expected, optimized_outputs):
            self.assertEqualArray(want, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
