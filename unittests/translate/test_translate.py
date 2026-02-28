import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.translate import translate, translate_header, Translater
from yobx.translate.inner_emitter import InnerEmitter
from yobx.translate.light_emitter import LightEmitter
from yobx.translate.make_helper import make_node_extended, make_ref_attribute


def _make_simple_model():
    """Creates a simple ONNX model: Y = Transpose(Reshape(X, [-1, 1]))."""
    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, None])
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None, None])
    shape_cst = onh.from_array(np.array([-1, 1], dtype=np.int64), name="shape")
    reshape = oh.make_node("Reshape", ["X", "shape"], ["reshaped"])
    transpose = oh.make_node("Transpose", ["reshaped"], ["Y"], perm=[1, 0])
    graph = oh.make_graph([reshape, transpose], "simple", [X], [Y], [shape_cst])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    return model


class TestTranslate(ExtTestCase):
    def test_translate_header_onnx(self):
        header = translate_header("onnx")
        self.assertIn("import onnx", header)
        self.assertIn("import onnx.helper as oh", header)
        self.assertIn("make_node_extended", header)

    def test_translate_header_light(self):
        header = translate_header("light")
        self.assertIn("onnx_array_api", header)

    def test_translate_header_builder(self):
        header = translate_header("builder")
        self.assertIn("GraphBuilder", header)

    def test_translate_header_invalid(self):
        self.assertRaise(lambda: translate_header("invalid"), ValueError)

    def test_translate_onnx(self):
        model = _make_simple_model()
        code = translate(model, api="onnx")
        self.assertIsInstance(code, str)
        self.assertIn("oh.make_model", code)
        self.assertIn("oh.make_graph", code)
        self.assertIn("Reshape", code)
        self.assertIn("Transpose", code)

    def test_translate_onnx_short(self):
        model = _make_simple_model()
        code = translate(model, api="onnx-short")
        self.assertIsInstance(code, str)
        self.assertIn("oh.make_model", code)

    def test_translate_light(self):
        model = _make_simple_model()
        code = translate(model, api="light")
        self.assertIsInstance(code, str)
        self.assertIn("start(", code)

    def test_translate_invalid_api(self):
        model = _make_simple_model()
        self.assertRaise(lambda: translate(model, api="unknown"), ValueError)

    def test_translater_inner_emitter(self):
        model = _make_simple_model()
        tr = Translater(model, emitter=InnerEmitter())
        code = tr.export(as_str=True)
        self.assertIsInstance(code, str)
        self.assertIn("opset_imports", code)

    def test_translater_light_emitter(self):
        model = _make_simple_model()
        tr = Translater(model, emitter=LightEmitter())
        code = tr.export(as_str=True)
        self.assertIsInstance(code, str)
        self.assertIn("start(", code)

    def test_translater_short_initializer(self):
        # Build a model with a large initializer (>16 elements to trigger short form)
        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [5, 5])
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [5, 5])
        big_w = onh.from_array(
            np.random.randn(5, 5).astype(np.float32), name="big_weight"
        )
        add = oh.make_node("Add", ["X", "big_weight"], ["Y"])
        graph = oh.make_graph([add], "big_model", [X], [Y], [big_w])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])

        code_full = translate(model, api="onnx")
        code_short = translate(model, api="onnx-short")
        # short version should not include the full array values
        self.assertLess(len(code_short), len(code_full))

    def test_make_node_extended_basic(self):
        node = make_node_extended("Add", ["X", "Y"], ["Z"])
        self.assertEqual(node.op_type, "Add")
        self.assertEqual(list(node.input), ["X", "Y"])
        self.assertEqual(list(node.output), ["Z"])

    def test_make_node_extended_with_domain(self):
        node = make_node_extended("MatMul", ["A", "B"], ["C"], domain="com.microsoft")
        self.assertEqual(node.domain, "com.microsoft")

    def test_make_node_extended_with_attrs(self):
        node = make_node_extended("Transpose", ["X"], ["Y"], perm=[1, 0])
        self.assertEqual(node.op_type, "Transpose")
        self.assertEqual(len(node.attribute), 1)

    def test_make_ref_attribute(self):
        att = make_ref_attribute("axis", 1, ref_attr_name="my_axis")
        self.assertEqual(att.name, "axis")
        self.assertEqual(att.ref_attr_name, "my_axis")

    def test_translate_code_is_executable(self):
        """Verify the generated 'onnx' code can be executed to recreate the model."""
        model = _make_simple_model()
        code = translate(model, api="onnx")
        header = translate_header("onnx")
        full_code = header + "\n" + code
        ns = {}
        exec(compile(full_code, "<string>", "exec"), ns)  # noqa: S102
        recreated = ns["model"]
        self.assertIsInstance(recreated, onnx.ModelProto)
        # Check the graph has the same number of nodes
        self.assertEqual(
            len(model.graph.node), len(recreated.graph.node)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
