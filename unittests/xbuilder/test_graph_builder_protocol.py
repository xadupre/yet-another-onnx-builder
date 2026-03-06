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

from yobx.builder.onnxscript import OnnxScriptGraphBuilder
from yobx.ext_test_case import ExtTestCase, requires_onnxscript
from yobx.typing import GraphBuilderProtocol, GraphBuilderExtendedProtocol, OpsetProtocol
from yobx.xbuilder import GraphBuilder

TFLOAT = TensorProto.FLOAT


class TestGraphBuilderProtocolExists(ExtTestCase):
    """Protocol class is importable from both yobx.typing and yobx.xbuilder."""

    def test_import_from_typing(self):
        self.assertIsNotNone(GraphBuilderProtocol)

    def test_import_from_xbuilder(self):
        from yobx.xbuilder import GraphBuilderProtocol as P

        self.assertIs(P, GraphBuilderProtocol)

    def test_extended_import_from_typing(self):
        self.assertIsNotNone(GraphBuilderExtendedProtocol)

    def test_extended_import_from_xbuilder(self):
        from yobx.xbuilder import GraphBuilderExtendedProtocol as P

        self.assertIs(P, GraphBuilderExtendedProtocol)

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
        g = OnnxScriptGraphBuilder(18)
        g.make_tensor_input("x", TFLOAT, (None, None))
        g.make_tensor_input("y", TFLOAT, (None, None))
        out = g.make_node("Add", ["x", "y"])
        g.make_tensor_output(out, TFLOAT, (None, None))
        return g

    def test_has_all_protocol_attributes(self):
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
        g = OnnxScriptGraphBuilder(18)
        g.make_tensor_input("x", TFLOAT, (None,))
        self.assertTrue(g.has_name("x"))
        self.assertFalse(g.has_name("does_not_exist"))

    def test_unique_name(self):
        g = OnnxScriptGraphBuilder(18)
        n1 = g.unique_name("tmp")
        n2 = g.unique_name("tmp")
        self.assertNotEqual(n1, n2)

    def test_get_opset(self):
        g = OnnxScriptGraphBuilder(18)
        self.assertEqual(g.get_opset(""), 18)

    def test_add_domain(self):
        g = OnnxScriptGraphBuilder(18)
        g.add_domain("custom", 1)
        self.assertEqual(g.get_opset("custom"), 1)

    def test_has_opset(self):
        g = OnnxScriptGraphBuilder(18)
        self.assertEqual(g.has_opset(""), 18)
        self.assertEqual(g.has_opset("custom.domain"), 0)

    def test_to_onnx(self):
        model = self._make_simple_model().to_onnx()
        self.assertIsNotNone(model)


class TestGraphBuilderExtendedProtocol(ExtTestCase):
    """GraphBuilder satisfies GraphBuilderExtendedProtocol."""

    EXTENDED_ATTRS = [
        "op",
        "set_type_shape_unary_op",
    ]

    def test_extended_protocol_has_required_methods(self):
        for name in self.EXTENDED_ATTRS:
            self.assertIn(
                name,
                dir(GraphBuilderExtendedProtocol),
                msg=f"GraphBuilderExtendedProtocol is missing '{name}'",
            )

    def test_extended_inherits_base(self):
        self.assertTrue(issubclass(GraphBuilderExtendedProtocol, GraphBuilderProtocol))

    def test_graphbuilder_has_extended_attrs(self):
        g = GraphBuilder(18, ir_version=9)
        for attr in self.EXTENDED_ATTRS:
            self.assertTrue(
                hasattr(g, attr),
                msg=f"GraphBuilder missing extended attribute '{attr}'",
            )

    def test_graphbuilder_is_instance_extended(self):
        g = GraphBuilder(18, ir_version=9)
        self.assertIsInstance(g, GraphBuilderExtendedProtocol)

    def test_set_type_shape_unary_op(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("x", TensorProto.FLOAT, (2, 3))
        g.make_node("Relu", ["x"], ["y"], name="Relu_0")
        g.set_type_shape_unary_op("y", "x")
        self.assertEqual(g.get_type("y"), TensorProto.FLOAT)
        self.assertEqual(g.get_shape("y"), (2, 3))

    def test_op_attribute_exists(self):
        g = GraphBuilder(18, ir_version=9)
        self.assertIsNotNone(g.op)


@requires_onnxscript()
class TestOnnxScriptGraphBuilderExtendedProtocol(ExtTestCase):
    """OnnxScriptGraphBuilder satisfies GraphBuilderExtendedProtocol."""

    def test_onnxscript_has_extended_attrs(self):
        g = OnnxScriptGraphBuilder(18)
        for attr in ["op", "set_type_shape_unary_op"]:
            self.assertTrue(
                hasattr(g, attr),
                msg=f"OnnxScriptGraphBuilder missing extended attribute '{attr}'",
            )

    def test_onnxscript_is_instance_extended(self):
        g = OnnxScriptGraphBuilder(18)
        self.assertIsInstance(g, GraphBuilderExtendedProtocol)


class TestOpsetProtocol(ExtTestCase):
    """OpsetProtocol is importable and has the required make_node method."""

    def test_opset_protocol_has_make_node(self):
        self.assertIn(
            "make_node",
            dir(OpsetProtocol),
            msg="OpsetProtocol is missing 'make_node'",
        )

    def test_graphbuilder_op_satisfies_protocol(self):
        g = GraphBuilder(18, ir_version=9)
        self.assertIsInstance(g.op, OpsetProtocol)

    def test_import_from_xbuilder(self):
        from yobx.xbuilder import OpsetProtocol as P

        self.assertIs(P, OpsetProtocol)

    def test_import_from_root(self):
        from yobx import OpsetProtocol as P

        self.assertIs(P, OpsetProtocol)


@requires_onnxscript()
class TestOnnxScriptOpsetProtocol(ExtTestCase):
    """OnnxScriptGraphBuilder's op helper satisfies OpsetProtocol."""

    def test_onnxscript_op_satisfies_protocol(self):
        g = OnnxScriptGraphBuilder(18)
        self.assertIsInstance(g.op, OpsetProtocol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
