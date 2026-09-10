import os
import unittest

from onnx_light import onnx
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light.onnx_core.optimization import GraphGraph

from yobx.ext_test_case import ExtTestCase


class TestGraphPatternOptimizationSequence(ExtTestCase):
    def test_sequences_split(self):
        data = os.path.join(os.path.dirname(__file__), "data", "sequences.onnx")
        model = onnx.load(data)
        builder = GraphBuilder(model)
        optimizer = GraphGraph(
            builder, patterns=["SplitToSequenceSequenceAt"], use_global_patterns=False
        )
        optimizer.optimize()
        optimized = builder.to_model(model.ir_version)
        op_types = [node.op_type for node in optimized.graph.node]
        self.assertIn("Split", op_types)
        self.assertNotIn("SplitToSequence", op_types)
        self.assertNotIn("SequenceAt", op_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
