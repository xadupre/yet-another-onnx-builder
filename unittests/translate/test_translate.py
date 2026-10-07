import unittest

import numpy as np
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto
from onnx_light.tools import translate, translate_header

from yobx.ext_test_case import ExtTestCase


def _make_simple_model():
    """Creates a simple ONNX model."""
    graph = oh.make_graph(
        [
            oh.make_node("Reshape", ["X", "shape"], ["reshaped"]),
            oh.make_node("Transpose", ["reshaped"], ["Y"], perm=[1, 0]),
        ],
        "simple",
        [oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])],
        [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])],
        [onh.from_array(np.array([-1, 1], dtype=np.int64), name="shape")],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)], ir_version=8)


class TestTranslate(ExtTestCase):
    def test_translate_header_compact(self):
        header = translate_header("onnx-compact")
        self.assertIn("import onnx_light.onnx as onnx", header)
        self.assertIn("import onnx_light.onnx.helper as oh", header)
        self.assertIn("import onnx_light.onnx.numpy_helper as onh", header)

    def test_translate_header_builder(self):
        header = translate_header("builder")
        self.assertIn("onnx_light.onnx_core.graph_builder", header)
        self.assertIn("GraphBuilder", header)

    def test_translate_header_invalid(self):
        self.assertRaise(lambda: translate_header("invalid"), ValueError)

    def test_translate_invalid_api(self):
        self.assertRaise(lambda: translate(_make_simple_model(), api="unknown"), ValueError)

    def test_translate_compact_is_executable(self):
        model = _make_simple_model()
        code = translate(model, api="onnx-compact")
        namespace = {}
        exec(compile(translate_header("onnx-compact") + code, "<string>", "exec"), namespace)
        recreated = namespace["model"]
        self.assertEqual(recreated.ir_version, model.ir_version)
        self.assertEqual(
            [node.op_type for node in recreated.graph.node],
            [node.op_type for node in model.graph.node],
        )
        np.testing.assert_array_equal(
            onh.to_array(recreated.graph.initializer[0]), onh.to_array(model.graph.initializer[0])
        )

    def test_translate_builder_is_executable(self):
        model = _make_simple_model()
        code = translate(model, api="builder")
        namespace = {}
        exec(compile(translate_header("builder") + code, "<string>", "exec"), namespace)
        recreated = namespace["model"]
        self.assertEqual(recreated.ir_version, model.ir_version)
        self.assertEqual(
            [node.op_type for node in recreated.graph.node],
            [node.op_type for node in model.graph.node],
        )

    def test_translate_unknown_dimensions(self):
        code = translate(_make_simple_model(), api="onnx-compact")
        self.assertIn("(None, None)", code)
        self.assertNotIn("('', '')", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
