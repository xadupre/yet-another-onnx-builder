import os
import unittest
from yobx.ext_test_case import ExtTestCase
from yobx.xbuilder.graph_builder import GraphBuilder


class TestGraphPatternOptimizationSequence(ExtTestCase):
    def test_sequences_split(self):
        data = os.path.join(os.path.dirname(__file__), "data", "sequences.onnx")
        g = GraphBuilder(data)
        g.optimize()
        opt = g.to_onnx()
        self.assertEqual(["Split"], [n.op_type for n in opt.graph.node])


if __name__ == "__main__":
    unittest.main(verbosity=2)
