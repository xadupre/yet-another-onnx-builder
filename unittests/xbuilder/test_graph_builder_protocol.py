"""
Tests for the :class:`~yobx.typing.GraphBuilderProtocol`.

Verifies that both :class:`~yobx.xbuilder.GraphBuilder` and
:class:`~yobx.builder.onnxscript.OnnxScriptGraphBuilder` satisfy the protocol
via ``isinstance`` checks (``runtime_checkable`` is *not* used for
:class:`GraphBuilderProtocol` because Protocol's structural check only covers
callable methods, not all signatures; instead the tests verify that each
required method/property exists with a functional smoke test).
"""

import unittest

from onnx import TensorProto

from yobx.ext_test_case import ExtTestCase, requires_onnxscript
from yobx.typing import GraphBuilderProtocol
from yobx.xbuilder import GraphBuilder


TFLOAT = TensorProto.FLOAT


class TestGraphBuilderProtocolExists(ExtTestCase):
    """Protocol class is importable from both yobx.typing and yobx.xbuilder."""

    def test_import_from_typing(self):
        from yobx.typing import GraphBuilderProtocol as P  # noqa: F401

        self.assertIsNotNone(P)

    def test_import_from_xbuilder(self):
        from yobx.xbuilder import GraphBuilderProtocol as P  # noqa: F401

        self.assertIsNotNone(P)

    def test_protocol_has_required_methods(self):
        required = [
            "input_names",
            "output_names",
            "get_opset",
            "add_domain",
            "has_opset",
            "unique_name",
            "has_name",
            "has_type",
            "get_type",
            "set_type",
            "has_shape",
            "get_shape",
            "set_shape",
            "make_tensor_input",
            "make_tensor_output",
            "make_initializer",
            "make_node",
            "to_onnx",
        ]
        for name in required:
            self.assertIn(
                name,
                dir(GraphBuilderProtocol),
                msg=f"GraphBuilderProtocol is missing '{name}'",
            )


class TestGraphBuilderSatisfiesProtocol(ExtTestCase):
    """GraphBuilder has all methods required by GraphBuilderProtocol."""

    def _make_simple_model(self):
        """Build a trivial Add graph and return the builder."""
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("x", TFLOAT, (None, None))
        g.make_tensor_input("y", TFLOAT, (None, None))
        g.make_node("Add", ["x", "y"], ["output_0"], name="Add_0")
        g.make_tensor_output("output_0", TFLOAT, (None, None))
        return g

    def test_has_all_protocol_attributes(self):
        g = self._make_simple_model()
        for attr in [
            "input_names",
            "output_names",
            "get_opset",
            "add_domain",
            "has_opset",
            "unique_name",
            "has_name",
            "has_type",
            "get_type",
            "set_type",
            "has_shape",
            "get_shape",
            "set_shape",
            "make_tensor_input",
            "make_tensor_output",
            "make_initializer",
            "make_node",
            "to_onnx",
        ]:
            self.assertTrue(hasattr(g, attr), msg=f"GraphBuilder missing '{attr}'")

    def test_input_output_names(self):
        g = self._make_simple_model()
        self.assertIn("x", g.input_names)
        self.assertIn("y", g.input_names)
        self.assertIn("output_0", g.output_names)

    def test_has_name(self):
        g = self._make_simple_model()
        self.assertTrue(g.has_name("x"))
        self.assertFalse(g.has_name("does_not_exist"))

    def test_unique_name(self):
        g = GraphBuilder(18, ir_version=9)
        n1 = g.unique_name("tmp")
        n2 = g.unique_name("tmp")
        self.assertNotEqual(n1, n2)

    def test_get_opset(self):
        g = GraphBuilder(18, ir_version=9)
        self.assertEqual(g.get_opset(""), 18)

    def test_has_opset(self):
        g = GraphBuilder(18, ir_version=9)
        self.assertEqual(g.has_opset(""), 18)
        self.assertEqual(g.has_opset("custom.domain"), 0)

    def test_graphbuilder_is_instance(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("x", TFLOAT, (None,))
        self.assertIsInstance(g, GraphBuilderProtocol)



    def test_to_onnx(self):
        g = self._make_simple_model()
        model = g.to_onnx()
        self.assertIsNotNone(model)


@requires_onnxscript()
class TestOnnxScriptGraphBuilderSatisfiesProtocol(ExtTestCase):
    """OnnxScriptGraphBuilder has all methods required by GraphBuilderProtocol."""

    def _make_simple_model(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        g.make_tensor_input("x", TFLOAT, (None, None))
        g.make_tensor_input("y", TFLOAT, (None, None))
        out = g.make_node("Add", ["x", "y"])
        g.make_tensor_output(out, TFLOAT, (None, None))
        return g

    def test_has_all_protocol_attributes(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        for attr in [
            "input_names",
            "output_names",
            "get_opset",
            "add_domain",
            "has_opset",
            "unique_name",
            "has_name",
            "has_type",
            "get_type",
            "set_type",
            "has_shape",
            "get_shape",
            "set_shape",
            "make_tensor_input",
            "make_tensor_output",
            "make_initializer",
            "make_node",
            "to_onnx",
        ]:
            self.assertTrue(
                hasattr(g, attr),
                msg=f"OnnxScriptGraphBuilder missing '{attr}'",
            )

    def test_input_output_names(self):
        g = self._make_simple_model()
        self.assertIn("x", g.input_names)
        self.assertIn("y", g.input_names)

    def test_has_name(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        g.make_tensor_input("x", TFLOAT, (None,))
        self.assertTrue(g.has_name("x"))
        self.assertFalse(g.has_name("does_not_exist"))

    def test_unique_name(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        n1 = g.unique_name("tmp")
        n2 = g.unique_name("tmp")
        self.assertNotEqual(n1, n2)

    def test_get_opset(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        self.assertEqual(g.get_opset(""), 18)

    def test_add_domain(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        g.add_domain("custom", 1)
        self.assertEqual(g.get_opset("custom"), 1)

    def test_has_opset(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        self.assertEqual(g.has_opset(""), 18)
        self.assertEqual(g.has_opset("custom.domain"), 0)

    def test_to_onnx(self):
        model = self._make_simple_model().to_onnx()
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
