import unittest

import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto
from onnx_light.tools import translate, translate_header

from yobx.ext_test_case import ExtTestCase


class TestReverseGraphBuilder(ExtTestCase):
    def test_builder_translation_uses_native_graph_builder(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "add",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, 3]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 3]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [None, 3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        code = translate(model, api="builder")
        self.assertIn("g = GraphBuilder('add')", code)
        self.assertIn(
            "from onnx_light.onnx_core.graph_builder import GraphBuilder",
            translate_header("builder"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
