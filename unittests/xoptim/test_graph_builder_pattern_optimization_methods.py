import unittest
from onnx import TensorProto
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase
from yobx.xbuilder.graph_builder import GraphBuilder, OptimizationOptions, InferShapesOptions
from yobx.xoptim import GraphBuilderPatternOptimization, EasyPatternOptimization
from yobx.xoptim.patterns_api import MatchResult, PatternOptimization, pattern_table_doc

TFLOAT = TensorProto.FLOAT
TINT64 = TensorProto.INT64

T = str


def _make_simple_gbpo(opset: int = 18):
    """Creates a simple GraphBuilderPatternOptimization from a tiny model."""
    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("Add", ["X", "Y"], ["Z"])],
            "test",
            [
                oh.make_tensor_value_info("X", TFLOAT, [3, 4]),
                oh.make_tensor_value_info("Y", TFLOAT, [3, 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [3, 4])],
        ),
        opset_imports=[oh.make_opsetid("", opset)],
        ir_version=9,
    )
    gr = GraphBuilder(
        model,
        infer_shapes_options=InferShapesOptions.BUILDER,
        optimization_options=OptimizationOptions(patterns=[]),
    )
    gro = GraphBuilderPatternOptimization(gr, patterns=[])
    return gr, gro


class TestGraphBuilderPatternOptimizationMethods(ExtTestCase):
    # ------------------------------------------------------------------
    # try_infer_type
    # ------------------------------------------------------------------

    def test_try_infer_type_known(self):
        """try_infer_type returns correct type for a name with known type."""
        _gr, gro = _make_simple_gbpo()
        # "X" should have type FLOAT since we defined it as such
        t = gro.try_infer_type("X")
        self.assertEqual(t, TensorProto.FLOAT)

    def test_try_infer_type_output(self):
        """try_infer_type returns type for output result."""
        _gr, gro = _make_simple_gbpo()
        t = gro.try_infer_type("Z")
        self.assertEqual(t, TensorProto.FLOAT)

    def test_try_infer_type_unknown_returns_zero(self):
        """try_infer_type returns 0 for an unknown name (no exc)."""
        _gr, gro = _make_simple_gbpo()
        t = gro.try_infer_type("nonexistent", exc=False)
        self.assertEqual(t, 0)

    # ------------------------------------------------------------------
    # try_infer_shape
    # ------------------------------------------------------------------

    def test_try_infer_shape_known(self):
        """try_infer_shape returns correct shape for a name with known shape."""
        _gr, gro = _make_simple_gbpo()
        shape = gro.try_infer_shape("X")
        self.assertEqual(shape, (3, 4))

    def test_try_infer_shape_output(self):
        """try_infer_shape returns shape for output result."""
        _gr, gro = _make_simple_gbpo()
        shape = gro.try_infer_shape("Z")
        self.assertEqual(shape, (3, 4))

    def test_try_infer_shape_unknown_returns_none(self):
        """try_infer_shape returns None for unknown name (no exc)."""
        _gr, gro = _make_simple_gbpo()
        shape = gro.try_infer_shape("nonexistent", exc=False)
        self.assertIsNone(shape)

    def test_try_infer_shape_unknown_exc(self):
        """try_infer_shape raises RuntimeError for unknown name when exc=True."""
        _, gro = _make_simple_gbpo()
        self.assertRaise(lambda: gro.try_infer_shape("nonexistent", exc=True), RuntimeError)

    # ------------------------------------------------------------------
    # make_node_check_opset
    # ------------------------------------------------------------------

    def test_make_node_check_opset_squeeze_ge13(self):
        """make_node_check_opset for Squeeze with opset >= 13 uses axes as input."""
        _, gro = _make_simple_gbpo(opset=18)
        node = gro.make_node_check_opset("Squeeze", ["X"], ["Z"], name="sq", axes=1)
        self.assertEqual(node.op_type, "Squeeze")
        # The axes should be an input (the second input), not an attribute
        self.assertGreaterEqual(len(node.input), 2)
        attr_names = {a.name for a in node.attribute}
        self.assertNotIn("axes", attr_names)

    def test_make_node_check_opset_unsqueeze_ge13(self):
        """make_node_check_opset for Unsqueeze with opset >= 13 uses axes as input."""
        _, gro = _make_simple_gbpo(opset=18)
        node = gro.make_node_check_opset("Unsqueeze", ["X"], ["Z"], name="usq", axes=0)
        self.assertEqual(node.op_type, "Unsqueeze")
        self.assertGreaterEqual(len(node.input), 2)
        attr_names = {a.name for a in node.attribute}
        self.assertNotIn("axes", attr_names)

    def test_make_node_check_opset_unsupported_op(self):
        """make_node_check_opset raises RuntimeError for unsupported op types."""
        _, gro = _make_simple_gbpo()
        self.assertRaise(
            lambda: gro.make_node_check_opset("MatMul", ["X", "Y"], ["Z"]), RuntimeError
        )

    def test_make_node_check_opset_wrong_domain(self):
        """make_node_check_opset raises AssertionError for non-default domain."""
        _, gro = _make_simple_gbpo()
        self.assertRaise(
            lambda: gro.make_node_check_opset("Squeeze", ["X"], ["Z"], domain="com.microsoft"),
            AssertionError,
        )

    # ------------------------------------------------------------------
    # _propagate_metadata
    # ------------------------------------------------------------------

    def test_propagate_metadata_basic(self):
        """_propagate_metadata copies metadata from old to new nodes."""
        _, gro = _make_simple_gbpo()

        old_node = oh.make_node("Add", ["X", "Y"], ["Z"])
        entry = old_node.metadata_props.add()
        entry.key = "custom_key"
        entry.value = "custom_value"

        new_node = oh.make_node("Add", ["X", "Y"], ["Z2"])
        gro._propagate_metadata([old_node], [new_node])

        keys = {m.key: m.value for m in new_node.metadata_props}
        self.assertIn("custom_key", keys)
        self.assertEqual(keys["custom_key"], "custom_value")

    def test_propagate_metadata_no_old_nodes(self):
        """_propagate_metadata with no old nodes does nothing."""
        _, gro = _make_simple_gbpo()
        new_node = oh.make_node("Add", ["X", "Y"], ["Z2"])
        gro._propagate_metadata([], [new_node])
        self.assertEqual(len(new_node.metadata_props), 0)

    def test_propagate_metadata_conflicting_values_are_dropped(self):
        """_propagate_metadata drops keys whose values differ across old nodes."""
        _, gro = _make_simple_gbpo()

        node1 = oh.make_node("Add", ["a", "b"], ["c"])
        e1 = node1.metadata_props.add()
        e1.key = "k"
        e1.value = "v1"

        node2 = oh.make_node("Mul", ["c", "d"], ["e"])
        e2 = node2.metadata_props.add()
        e2.key = "k"
        e2.value = "v2"

        new_node = oh.make_node("Relu", ["a"], ["r"])
        gro._propagate_metadata([node1, node2], [new_node])

        keys = {m.key for m in new_node.metadata_props}
        self.assertNotIn("k", keys)

    def test_propagate_metadata_skips_type_shape_keys(self):
        """_propagate_metadata skips reserved type/shape metadata keys."""
        _, gro = _make_simple_gbpo()
        old_node = oh.make_node("Add", ["X", "Y"], ["Z"])
        for key in ("intypes", "outtypes", "inshapes", "outshapes"):
            e = old_node.metadata_props.add()
            e.key = key
            e.value = "something"

        new_node = oh.make_node("Add", ["X", "Y"], ["Z2"])
        gro._propagate_metadata([old_node], [new_node])

        keys = {m.key for m in new_node.metadata_props}
        for key in ("intypes", "outtypes", "inshapes", "outshapes"):
            self.assertNotIn(key, keys)

    def test_propagate_metadata_none_old_nodes_skipped(self):
        """_propagate_metadata skips None entries in old_nodes."""
        _, gro = _make_simple_gbpo()
        old_node = oh.make_node("Add", ["X", "Y"], ["Z"])
        e = old_node.metadata_props.add()
        e.key = "mykey"
        e.value = "myval"

        new_node = oh.make_node("Add", ["X", "Y"], ["Z2"])
        gro._propagate_metadata([None, old_node], [new_node])

        keys = {m.key: m.value for m in new_node.metadata_props}
        self.assertIn("mykey", keys)

    # ------------------------------------------------------------------
    # _chech_graph_verifies
    # ------------------------------------------------------------------

    def test_chech_graph_verifies_valid_matmul(self):
        """_chech_graph_verifies passes for a MatMul with compatible shapes."""
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MatMul", ["X", "Y"], ["Z"])],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 3]),
                    oh.make_tensor_value_info("Y", TFLOAT, [3, 4]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 4])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=[]),
        )
        gro = GraphBuilderPatternOptimization(gr, patterns=[])
        matmul_node = gr.nodes[0]
        # Should not raise
        gro._chech_graph_verifies(matmul_node)

    def test_chech_graph_verifies_invalid_matmul(self):
        """_chech_graph_verifies raises AssertionError for incompatible MatMul shapes."""
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MatMul", ["X", "Y"], ["Z"])],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 3]),
                    oh.make_tensor_value_info("Y", TFLOAT, [5, 4]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 4])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=[]),
        )
        gro = GraphBuilderPatternOptimization(gr, patterns=[])
        matmul_node = gr.nodes[0]
        self.assertRaise(lambda: gro._chech_graph_verifies(matmul_node), AssertionError)

    def test_chech_graph_verifies_non_matmul(self):
        """_chech_graph_verifies does nothing for non-MatMul/Gemm nodes."""
        _, gro = _make_simple_gbpo()
        add_node = oh.make_node("Add", ["X", "Y"], ["Z"])
        # Should not raise for Add
        gro._chech_graph_verifies(add_node)

    # ------------------------------------------------------------------
    # _assert_check_graph_nodes_
    # ------------------------------------------------------------------

    def test_assert_check_graph_nodes_raises(self):
        """_assert_check_graph_nodes_ always raises AssertionError."""
        _, gro = _make_simple_gbpo()
        node = oh.make_node("Add", ["X", "Y"], ["Z"])

        self.assertRaise(
            lambda: gro._assert_check_graph_nodes_(
                added_nodes=None,
                removed_nodes=None,
                nodes=[node],
                statistics=None,
                node=node,
                step="test_step",
                i="X",
                p=0,
            ),
            AssertionError,
        )

    # ------------------------------------------------------------------
    # optimize_node_subgraphs_inplace is tested via GraphBuilder in
    # unittests/xbuilder/test_graph_builder.py::test_optimize_node_subgraphs_inplace
    # ------------------------------------------------------------------


class TestMatchResultDebugString(ExtTestCase):
    # ------------------------------------------------------------------
    # MatchResult.debug_string
    # ------------------------------------------------------------------

    def test_debug_string_without_graph(self):
        """MatchResult.debug_string without a graph just shows node ops."""
        pat = PatternOptimization(verbose=0)
        node = oh.make_node("Add", ["X", "Y"], ["Z"])
        mr = MatchResult(pat, [node], lambda g, n: [])
        s = mr.debug_string()
        self.assertIn("Add", s)

    def test_debug_string_with_none_node(self):
        """MatchResult.debug_string handles None entries in the node list."""
        pat = PatternOptimization(verbose=0)
        node = oh.make_node("Relu", ["X"], ["Z"])
        mr = MatchResult(pat, [None, node], lambda g, n: [])
        s = mr.debug_string()
        self.assertIn("Relu", s)

    def test_debug_string_with_graph(self):
        """MatchResult.debug_string with a GraphBuilder shows shape/type info."""
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Z"])],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, [3, 4])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 4])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=[]),
        )
        pat = PatternOptimization(verbose=0)
        node = gr.nodes[0]
        mr = MatchResult(pat, [node], lambda g, n: [])
        s = mr.debug_string(g=gr)
        self.assertIn("Relu", s)
        self.assertIn("--------", s)


class TestPatternDebugPrint(ExtTestCase):
    # ------------------------------------------------------------------
    # PatternOptimization._debug_print
    # ------------------------------------------------------------------

    def test_base_debug_print_returns_empty(self):
        pat = PatternOptimization(verbose=0)
        self.assertEqual(pat._debug_print(), "")

    # ------------------------------------------------------------------
    # EasyPatternOptimization._debug_print
    # ------------------------------------------------------------------

    def test_easy_debug_print_without_debug_attr(self):
        class SimpleAddPattern(EasyPatternOptimization):
            """Simple add pattern."""

            def match_pattern(self, g, x: T, y: T):
                return g.op.Add(x, y)

            def apply_pattern(self, g, x: T, y: T):
                return g.op.Add(x, y)

        pat = SimpleAddPattern(verbose=0)
        self.assertEqual(pat._debug_print(), "")

    def test_easy_debug_print_with_debug_attr(self):
        class SimpleAddPattern(EasyPatternOptimization):
            """Simple add pattern."""

            def match_pattern(self, g, x: T, y: T):
                return g.op.Add(x, y)

            def apply_pattern(self, g, x: T, y: T):
                return g.op.Add(x, y)

        pat = SimpleAddPattern(verbose=0)
        node = oh.make_node("Add", ["x", "y"], ["z"])
        pat._debug = {"iteration": 1, "stacked": [1, 2], "marked": {0: (node, node)}}
        result = pat._debug_print()
        self.assertIsInstance(result, str)


class TestGetApplyPattern(ExtTestCase):
    # ------------------------------------------------------------------
    # EasyPatternOptimization._get_apply_pattern
    # ------------------------------------------------------------------

    def test_get_apply_pattern(self):
        class AddPattern(EasyPatternOptimization):
            def match_pattern(self, g, x: T, y: T):
                return g.op.Add(x, y)

            def apply_pattern(self, g, x: T, y: T):
                return g.op.Add(x, y)

        pat = AddPattern(verbose=0)
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("Y", TFLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=[]),
        )
        gro = GraphBuilderPatternOptimization(gr, patterns=[pat])
        apply_pat = pat._get_apply_pattern(gro)
        self.assertIsInstance(apply_pat, GraphBuilderPatternOptimization)

    def test_get_apply_pattern_is_cached(self):
        class AddPattern(EasyPatternOptimization):
            """Replaces Add(x, y) with Add(x, y)."""

            def match_pattern(self, g, x: T, y: T):
                return g.op.Add(x, y)

            def apply_pattern(self, g, x: T, y: T):
                return g.op.Add(x, y)

        pat = AddPattern(verbose=0)
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("Y", TFLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=[]),
        )
        gro = GraphBuilderPatternOptimization(gr, patterns=[pat])
        p1 = pat._get_apply_pattern(gro)
        p2 = pat._get_apply_pattern(gro)
        self.assertIs(p1, p2)


class TestPatternTableDoc(ExtTestCase):
    # ------------------------------------------------------------------
    # pattern_table_doc
    # ------------------------------------------------------------------

    def test_pattern_table_doc_returns_list(self):
        from yobx.xoptim import get_pattern_list

        patterns = get_pattern_list("default")[:3]
        result = pattern_table_doc(patterns, as_rst=False)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        for item in result:
            self.assertIn("name", item)
            self.assertIn("priority", item)
            self.assertIn("doc", item)

    def test_pattern_table_doc_fields(self):
        from yobx.xoptim import get_pattern_list

        patterns = get_pattern_list("default")[:1]
        result = pattern_table_doc(patterns, as_rst=False)
        self.assertEqual(len(result), 1)
        item = result[0]
        # name should match the class name
        self.assertEqual(item["name"], patterns[0].__class__.__name__)
        # short_name should be name without "Pattern" suffix
        self.assertEqual(item["short_name"], item["name"].replace("Pattern", ""))


def _make_mixed_gbpo(opset: int = 18):
    """Creates an optimizer from a graph with Add, Relu and Mul nodes."""
    from yobx.xbuilder.graph_builder import InferShapesOptions

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["add_out"], name="n_add"),
                oh.make_node("Relu", ["add_out"], ["relu_out"], name="n_relu"),
                oh.make_node("Mul", ["relu_out", "Y"], ["Z"], name="n_mul"),
            ],
            "test",
            [
                oh.make_tensor_value_info("X", TFLOAT, [3, 4]),
                oh.make_tensor_value_info("Y", TFLOAT, [3, 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [3, 4])],
        ),
        opset_imports=[oh.make_opsetid("", opset)],
        ir_version=9,
    )
    gr = GraphBuilder(
        model,
        infer_shapes_options=InferShapesOptions.BUILDER,
        optimization_options=OptimizationOptions(patterns=[]),
    )
    gro = GraphBuilderPatternOptimization(gr, patterns=[])
    gro._build()
    return gr, gro


class TestPatternOpTypesFiltering(ExtTestCase):
    """Tests for the op_types class attribute and its use in enumerate_matches."""

    # ------------------------------------------------------------------
    # Default value
    # ------------------------------------------------------------------

    def test_op_types_default_is_none(self):
        """PatternOptimization.op_types is None by default."""
        self.assertIsNone(PatternOptimization.op_types)

    # ------------------------------------------------------------------
    # PatternOptimization with op_types set – only matching nodes visited
    # ------------------------------------------------------------------

    def test_enumerate_matches_skips_non_matching_op_types(self):
        """When op_types is set, enumerate_matches only calls match() for nodes
        whose op_type is in op_types."""
        visited = []

        class AddOnlyPattern(PatternOptimization):
            op_types = {"Add"}

            def match(self, g, node, matched):
                visited.append(node.op_type)
                return None  # never actually match

        _, gro = _make_mixed_gbpo()
        pat = AddOnlyPattern()
        list(pat.enumerate_matches(gro))
        # Only Add nodes should have been visited.
        self.assertEqual(visited, ["Add"])

    def test_enumerate_matches_none_op_types_visits_all(self):
        """When op_types is None, enumerate_matches visits every node."""
        visited = []

        class AllPattern(PatternOptimization):
            op_types = None

            def match(self, g, node, matched):
                visited.append(node.op_type)
                return None

        _, gro = _make_mixed_gbpo()
        pat = AllPattern()
        list(pat.enumerate_matches(gro))
        self.assertEqual(sorted(visited), ["Add", "Mul", "Relu"])

    # ------------------------------------------------------------------
    # EasyPatternOptimization – op_types auto-inferred from the pattern
    # ------------------------------------------------------------------

    def test_easy_pattern_infers_op_types_from_pattern(self):
        """EasyPatternOptimization infers op_types from the anchor node of the
        match pattern so that enumerate_matches can skip irrelevant nodes."""

        class ReluPattern(EasyPatternOptimization):
            def match_pattern(self, g, x: T):
                return g.op.Relu(x)

            def apply_pattern(self, g, x: T):
                return g.op.Relu(x)

        _, gro = _make_mixed_gbpo()
        pat = ReluPattern()
        # Exhaust the iterator so that enumerate_matches runs fully.
        list(pat.enumerate_matches(gro))
        # After the first enumeration, op_types should have been inferred.
        self.assertIsNotNone(pat.op_types)
        self.assertIn("Relu", pat.op_types)

    def test_easy_pattern_explicit_op_types_not_overwritten(self):
        """When a subclass of EasyPatternOptimization explicitly sets op_types,
        that value must not be overwritten by the lazy inference."""

        class MulPattern(EasyPatternOptimization):
            op_types = {"Mul"}  # explicit – should remain unchanged

            def match_pattern(self, g, x: T, y: T):
                return g.op.Add(x, y)  # anchor is Add, but op_types says Mul

            def apply_pattern(self, g, x: T, y: T):
                return g.op.Add(x, y)

        _, gro = _make_mixed_gbpo()
        pat = MulPattern()
        list(pat.enumerate_matches(gro))
        # The explicit op_types should be preserved.
        self.assertEqual(pat.op_types, {"Mul"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
