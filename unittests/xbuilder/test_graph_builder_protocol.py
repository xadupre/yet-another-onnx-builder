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
from yobx.typing import (
    GraphBuilderProtocol,
    GraphBuilderExtendedProtocol,
    GraphBuilderTorchProtocol,
    OpsetProtocol,
)
from yobx.ext_test_case import ExtTestCase, requires_onnxscript
from yobx.xbuilder import GraphBuilder

TFLOAT = TensorProto.FLOAT


class TestGraphBuilderProtocolExists(ExtTestCase):
    """Protocol class is importable from both yobx.typing and yobx.xbuilder."""

    def test_import_from_typing(self):
        self.assertIsNotNone(GraphBuilderProtocol)

    def test_import_from_xbuilder(self):
        self.assertIs(GraphBuilderProtocol, GraphBuilderProtocol)

    def test_extended_import_from_typing(self):
        self.assertIsNotNone(GraphBuilderExtendedProtocol)

    def test_extended_import_from_xbuilder(self):
        self.assertIs(GraphBuilderExtendedProtocol, GraphBuilderExtendedProtocol)

    def test_protocol_has_required_methods(self):
        required = [
            "input_names",
            "output_names",
            "get_opset",
            "set_opset",
            "has_opset",
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
                name, dir(GraphBuilderProtocol), msg=f"GraphBuilderProtocol is missing '{name}'"
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
            "set_opset",
            "has_opset",
            "unique_name",
            "prefix_name_context",
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

    def test_prefix_name_context_prefixes_unique_name(self):
        g = GraphBuilder(18, ir_version=9)
        with g.prefix_name_context("mystep"):
            name = g.unique_name("var")
        self.assertEqual(name, "mystep__var")

    def test_prefix_name_context_restores_after_exit(self):
        g = GraphBuilder(18, ir_version=9)
        with g.prefix_name_context("mystep"):
            pass
        name = g.unique_name("var")
        self.assertEqual(name, "var")

    def test_prefix_name_context_nested(self):
        g = GraphBuilder(18, ir_version=9)
        with g.prefix_name_context("outer"), g.prefix_name_context("inner"):
            name = g.unique_name("var")
        self.assertEqual(name, "outer__inner__var")

    def test_prefix_name_context_restores_on_exception(self):
        g = GraphBuilder(18, ir_version=9)
        try:
            with g.prefix_name_context("mystep"):
                raise RuntimeError("test")
        except RuntimeError:
            pass
        name = g.unique_name("var")
        self.assertEqual(name, "var")

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
            "set_opset",
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
            self.assertTrue(hasattr(g, attr), msg=f"OnnxScriptGraphBuilder missing '{attr}'")

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

    def test_set_opset(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        g.set_opset("custom", 1)
        self.assertEqual(g.get_opset("custom"), 1)

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


class TestGraphBuilderExtendedProtocol(ExtTestCase):
    """GraphBuilder satisfies GraphBuilderExtendedProtocol."""

    EXTENDED_ATTRS = ["main_opset", "unique_name", "op", "set_type_shape_unary_op"]

    def test_extended_protocol_has_required_methods(self):
        for name in self.EXTENDED_ATTRS:
            self.assertIn(
                name,
                dir(GraphBuilderExtendedProtocol),
                msg=f"GraphBuilderExtendedProtocol is missing '{name}'",
            )

    def test_graphbuilder_has_extended_attrs(self):
        g = GraphBuilder(18, ir_version=9)
        for attr in self.EXTENDED_ATTRS:
            self.assertTrue(
                hasattr(g, attr), msg=f"GraphBuilder missing extended attribute '{attr}'"
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
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        for attr in ["main_opset", "op", "set_type_shape_unary_op"]:
            self.assertTrue(
                hasattr(g, attr),
                msg=f"OnnxScriptGraphBuilder missing extended attribute '{attr}'",
            )

    def test_onnxscript_is_instance_extended(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        self.assertIsInstance(g, GraphBuilderExtendedProtocol)


class TestOpsetProtocol(ExtTestCase):
    """OpsetProtocol is importable and has the required __getattr__ method."""

    def test_opset_protocol_has_getattr(self):
        self.assertIn(
            "__getattr__", dir(OpsetProtocol), msg="OpsetProtocol is missing '__getattr__'"
        )

    def test_graphbuilder_op_satisfies_protocol(self):
        g = GraphBuilder(18, ir_version=9)
        self.assertIsInstance(g.op, OpsetProtocol)

    def test_import_from_xbuilder(self):
        self.assertIs(OpsetProtocol, OpsetProtocol)

    def test_import_from_root(self):
        self.assertIs(OpsetProtocol, OpsetProtocol)


@requires_onnxscript()
class TestOnnxScriptOpsetProtocol(ExtTestCase):
    """OnnxScriptGraphBuilder's op helper satisfies OpsetProtocol."""

    def test_onnxscript_op_satisfies_protocol(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        self.assertIsInstance(g.op, OpsetProtocol)


class TestGraphBuilderTorchProtocol(ExtTestCase):
    """GraphBuilderTorchProtocol is importable and has the expected members."""

    # Methods / properties added beyond GraphBuilderExtendedProtocol.
    TORCH_ONLY_ATTRS = [
        # rank helpers
        "has_rank",
        "get_rank",
        "set_rank",
        # device helpers
        "has_device",
        "get_device",
        "set_device",
        # extended type / shape
        "get_type_known",
        "set_shapes_types",
        # sequence support
        "is_sequence",
        "get_sequence",
        "set_sequence",
        "make_tensor_sequence_input",
        # dynamic-shape helpers
        "is_dynamic_shape",
        "get_input_dynamic_shape",
        "verify_dynamic_shape",
        "register_dynamic_objects_from_shape",
        "make_dynamic_object",
        "add_dynamic_object",
        "make_new_dynamic_shape",
        # sub-builder / local functions
        "make_nodes",
        "make_local_function",
        "make_subset_builder",
        # misc methods
        "add_stat",
        "pretty_text",
        "register_users",
        "extract_input_names_from_args",
        # state attributes / properties
        "anyop",
        "last_added_node",
    ]

    def test_import_from_typing(self):
        self.assertIsNotNone(GraphBuilderTorchProtocol)

    def test_import_from_xbuilder(self):
        from yobx.xbuilder import GraphBuilderTorchProtocol as T

        self.assertIs(T, GraphBuilderTorchProtocol)

    def test_is_subprotocol_of_extended(self):
        # GraphBuilderTorchProtocol must list GraphBuilderExtendedProtocol in
        # its MRO.  We cannot use issubclass() for runtime-checkable protocols
        # that declare data attributes (Python raises TypeError), so we inspect
        # the MRO directly.
        self.assertIn(GraphBuilderExtendedProtocol, GraphBuilderTorchProtocol.__mro__)

    def test_protocol_has_torch_attrs(self):
        for name in self.TORCH_ONLY_ATTRS:
            self.assertIn(
                name,
                dir(GraphBuilderTorchProtocol),
                msg=f"GraphBuilderTorchProtocol is missing '{name}'",
            )

    def test_graphbuilder_has_torch_attrs(self):
        g = GraphBuilder(18, ir_version=9)
        for name in self.TORCH_ONLY_ATTRS:
            self.assertTrue(hasattr(g, name), msg=f"GraphBuilder missing attribute '{name}'")

    def test_graphbuilder_is_instance_torch_protocol(self):
        g = GraphBuilder(18, ir_version=9)
        self.assertIsInstance(g, GraphBuilderTorchProtocol)

    def test_has_rank_returns_false_for_unknown(self):
        g = GraphBuilder(18, ir_version=9)
        self.assertFalse(g.has_rank("nonexistent"))

    def test_set_get_rank(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("x", TensorProto.FLOAT, (2, 3))
        g.set_rank("x", 2)
        self.assertTrue(g.has_rank("x"))
        self.assertEqual(g.get_rank("x"), 2)

    def test_is_sequence_false_for_regular_tensor(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("x", TensorProto.FLOAT, (2,))
        self.assertFalse(g.is_sequence("x"))

    def test_anyop_attribute_exists(self):
        g = GraphBuilder(18, ir_version=9)
        self.assertIsNotNone(g.anyop)

    def test_last_added_node_none_on_empty_graph(self):
        g = GraphBuilder(18, ir_version=9)
        self.assertIsNone(g.last_added_node)


if __name__ == "__main__":
    unittest.main(verbosity=2)
