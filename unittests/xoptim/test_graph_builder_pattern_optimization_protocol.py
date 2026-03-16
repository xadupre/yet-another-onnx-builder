"""
Tests for :class:`~yobx.typing.GraphBuilderPatternOptimizationProtocol`.

Verifies that :class:`~yobx.xoptim.GraphBuilderPatternOptimization` satisfies
the protocol and that the protocol exposes all the methods pattern authors
depend on.
"""

import unittest
from onnx import TensorProto
from yobx.typing import GraphBuilderPatternOptimizationProtocol
from yobx.ext_test_case import ExtTestCase
from yobx.xbuilder import GraphBuilder
from yobx.xoptim import GraphBuilderPatternOptimization

TFLOAT = TensorProto.FLOAT


def _make_optimizer() -> GraphBuilderPatternOptimization:
    """Build a trivial Add graph and return its pattern optimizer."""
    g = GraphBuilder(18, ir_version=9)
    g.make_tensor_input("x", TFLOAT, (None, None))
    g.make_tensor_input("y", TFLOAT, (None, None))
    g.make_node("Add", ["x", "y"], ["output_0"], name="Add_0")
    g.make_tensor_output("output_0", TFLOAT, (None, None))
    return GraphBuilderPatternOptimization(g)


class TestGraphBuilderPatternOptimizationProtocolImport(ExtTestCase):
    """Protocol is importable from yobx.typing."""

    def test_import_from_typing(self):
        self.assertIsNotNone(GraphBuilderPatternOptimizationProtocol)

    def test_protocol_is_runtime_checkable(self):
        # Importing typing.runtime_checkable ensures isinstance() works.
        from typing import runtime_checkable

        self.assertTrue(
            isinstance(GraphBuilderPatternOptimizationProtocol, type(runtime_checkable))
            or hasattr(GraphBuilderPatternOptimizationProtocol, "_is_protocol")
        )


class TestGraphBuilderPatternOptimizationProtocolMembers(ExtTestCase):
    """Protocol defines all members that pattern authors use."""

    REQUIRED_MEMBERS = [
        # instance attributes / properties
        "verbose",
        "processor",
        "builder",
        # graph-level properties
        "main_opset",
        "opsets",
        "nodes",
        "input_names",
        "output_names",
        "inputs",
        "outputs",
        # node navigation
        "iter_nodes",
        "node_before",
        "next_node",
        "next_nodes",
        "get_position",
        # liveness
        "is_used",
        "is_used_more_than_once",
        "is_used_only_by",
        "is_output",
        "is_used_by_subgraph",
        # constant queries
        "is_constant",
        "is_constant_scalar",
        "get_constant_shape",
        "get_computed_constant",
        "get_constant_scalar",
        "get_constant_or_attribute",
        # type / shape queries
        "has_type",
        "get_type",
        "has_rank",
        "get_rank",
        "has_shape",
        "get_shape",
        "same_shape",
        "get_shape_renamed",
        "try_infer_type",
        "try_infer_shape",
        # attribute helpers
        "get_attribute",
        "get_attribute_with_default",
        "get_attributes_with_default",
        "get_axis",
        # processor / constraint helpers
        "has_processor",
        "get_registered_constraints",
        "has_exact_same_constant_in_context",
        "do_not_turn_constant_initializers_maybe_because_of_showing",
        # creation
        "make_initializer",
        "unique_name",
        "make_node",
        "make_node_check_opset",
        # misc
        "pretty_text",
    ]

    def test_protocol_has_required_members(self):
        proto_dir = dir(GraphBuilderPatternOptimizationProtocol)
        for name in self.REQUIRED_MEMBERS:
            self.assertIn(
                name,
                proto_dir,
                msg=f"GraphBuilderPatternOptimizationProtocol is missing '{name}'",
            )


class TestGraphBuilderPatternOptimizationSatisfiesProtocol(ExtTestCase):
    """GraphBuilderPatternOptimization satisfies the protocol."""

    def test_isinstance(self):
        go = _make_optimizer()
        self.assertIsInstance(go, GraphBuilderPatternOptimizationProtocol)

    def test_has_all_protocol_attributes(self):
        go = _make_optimizer()
        for attr in TestGraphBuilderPatternOptimizationProtocolMembers.REQUIRED_MEMBERS:
            self.assertTrue(
                hasattr(go, attr), msg=f"GraphBuilderPatternOptimization is missing '{attr}'"
            )

    def test_main_opset(self):
        go = _make_optimizer()
        self.assertEqual(go.main_opset, 18)

    def test_input_output_names(self):
        go = _make_optimizer()
        self.assertIn("x", go.input_names)
        self.assertIn("y", go.input_names)
        self.assertIn("output_0", go.output_names)

    def test_nodes(self):
        go = _make_optimizer()
        node_types = [n.op_type for n in go.nodes]
        self.assertIn("Add", node_types)

    def test_opsets(self):
        go = _make_optimizer()
        self.assertIn("", go.opsets)
        self.assertEqual(go.opsets[""], 18)

    def test_verbose_and_processor(self):
        go = _make_optimizer()
        self.assertIsInstance(go.verbose, int)
        self.assertIsInstance(go.processor, str)

    def test_builder_attribute(self):
        go = _make_optimizer()
        self.assertIsNotNone(go.builder)

    def test_is_used(self):
        go = _make_optimizer()
        self.assertTrue(go.is_used("output_0"))
        self.assertFalse(go.is_used("nonexistent_name"))

    def test_is_constant(self):
        go = _make_optimizer()
        self.assertFalse(go.is_constant("x"))

    def test_has_type_get_type(self):
        go = _make_optimizer()
        self.assertTrue(go.has_type("x"))
        self.assertEqual(go.get_type("x"), TFLOAT)

    def test_has_shape_get_shape(self):
        go = _make_optimizer()
        self.assertTrue(go.has_shape("x"))
        shape = go.get_shape("x")
        self.assertEqual(len(shape), 2)

    def test_has_rank_get_rank(self):
        go = _make_optimizer()
        self.assertTrue(go.has_rank("x"))
        self.assertEqual(go.get_rank("x"), 2)

    def test_node_before(self):
        go = _make_optimizer()
        # "x" is an input, so no node produces it
        self.assertIsNone(go.node_before("x"))
        # "output_0" is produced by Add
        node = go.node_before("output_0")
        self.assertIsNotNone(node)
        self.assertEqual(node.op_type, "Add")

    def test_next_nodes(self):
        go = _make_optimizer()
        # "x" is consumed by Add
        consumers = go.next_nodes("x")
        self.assertTrue(len(consumers) >= 1)
        self.assertEqual(consumers[0].op_type, "Add")

    def test_is_output(self):
        go = _make_optimizer()
        self.assertTrue(go.is_output("output_0"))
        self.assertFalse(go.is_output("x"))

    def test_unique_name(self):
        go = _make_optimizer()
        n1 = go.unique_name("tmp")
        n2 = go.unique_name("tmp")
        self.assertNotEqual(n1, n2)

    def test_has_processor(self):
        go = _make_optimizer()
        self.assertTrue(go.has_processor("CPU"))

    def test_pretty_text(self):
        go = _make_optimizer()
        text = go.pretty_text()
        self.assertIsInstance(text, str)
        self.assertIn("Add", text)

    def test_iter_nodes(self):
        go = _make_optimizer()
        nodes = list(go.iter_nodes())
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].op_type, "Add")

    def test_get_registered_constraints(self):
        go = _make_optimizer()
        constraints = go.get_registered_constraints()
        self.assertIsInstance(constraints, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
