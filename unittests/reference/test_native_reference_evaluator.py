import importlib.abc
import sys
import unittest
import numpy as np
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator
from yobx.reference import ExtendedReferenceEvaluator
from yobx.reference.ops._native_op import NativeOpKernel


class RejectLegacyRuntimeImports(importlib.abc.MetaPathFinder):
    """Rejects attempts to load a legacy model package or an alternate runtime."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"onnx", "onnxruntime"}:
            raise AssertionError(f"Native evaluation attempted to import {fullname!r}.")
        return None


class TestNativeReferenceEvaluator(unittest.TestCase):
    @staticmethod
    def make_model():
        """Builds an Add/MatMul graph with ordered outputs."""
        return helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Add", ["X", "B"], ["sum"]),
                    helper.make_node("MatMul", ["sum", "W"], ["product"]),
                ],
                "native_add_matmul",
                [
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2]),
                    helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2]),
                ],
                [
                    helper.make_tensor_value_info("product", TensorProto.FLOAT, [2, 2]),
                    helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2, 2]),
                ],
                initializer=[helper.make_tensor("B", TensorProto.FLOAT, [2], [1.0, -1.0])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )

    def setUp(self):
        self.import_guard = RejectLegacyRuntimeImports()
        sys.meta_path.insert(0, self.import_guard)
        self.feeds = {
            "X": np.arange(4, dtype=np.float32).reshape(2, 2),
            "W": np.array([[1, 2], [-1, 3]], dtype=np.float32),
        }
        self.expected_sum = self.feeds["X"] + np.array([1, -1], dtype=np.float32)
        self.expected_product = self.expected_sum @ self.feeds["W"]

    def tearDown(self):
        sys.meta_path.remove(self.import_guard)

    def test_native_model_and_serialization(self):
        from yobx.container import ExportArtifact

        model = self.make_model()
        for proto in (model, model.SerializeToString(), ExportArtifact(proto=model)):
            with self.subTest(serialized=isinstance(proto, bytes)):
                session = ExtendedReferenceEvaluator(proto)
                self.assertIsInstance(session, ReferenceEvaluator)
                self.assertEqual(session.input_names, ["X", "W"])
                self.assertEqual(session.output_names, ["product", "sum"])
                product, total = session.run(None, self.feeds)
                np.testing.assert_array_equal(product, self.expected_product)
                np.testing.assert_array_equal(total, self.expected_sum)
                np.testing.assert_array_equal(
                    session.run(["sum"], self.feeds)[0], self.expected_sum
                )
                np.testing.assert_array_equal(
                    session.run(list(self.feeds.values()))[0], self.expected_product
                )

    def test_graph_and_intermediate_outputs(self):
        session = ExtendedReferenceEvaluator(self.make_model().graph, opsets={"": 18})
        values = session.run(None, self.feeds, intermediate=True)
        self.assertEqual(set(values), {"", "X", "W", "B", "sum", "product"})
        np.testing.assert_array_equal(values["sum"], self.expected_sum)
        np.testing.assert_array_equal(values["product"], self.expected_product)
        self.assertEqual(len(session.input_types), 2)
        self.assertEqual(len(session.output_types), 2)

    def test_model_opset_override_preserves_model(self):
        model = self.make_model()
        original = model.SerializeToString()
        session = ExtendedReferenceEvaluator(model, opsets={"": 19})
        self.assertEqual(session.opsets, {"": 19})
        np.testing.assert_array_equal(session.run(None, self.feeds)[0], self.expected_product)
        self.assertEqual(model.SerializeToString(), original)

    def test_node_proto_repeated_and_optional_inputs(self):
        x = self.feeds["X"]
        node = helper.make_node("Add", ["X", "X"], ["Y"])
        session = ExtendedReferenceEvaluator(node)
        self.assertIs(session.proto_, node)
        self.assertEqual(session.input_names, ["X"])
        np.testing.assert_array_equal(session.run([x])[0], x * 2)
        intermediates = session.run(None, {"X": x}, intermediate=True)
        np.testing.assert_array_equal(intermediates["Y"], x * 2)
        self.assertEqual(len(session.input_types), 1)
        self.assertEqual(len(session.output_types), 1)

        session = ExtendedReferenceEvaluator(helper.make_node("Clip", ["X", "", "max"], ["Y"]))
        self.assertEqual(session.input_names, ["X", "max"])
        result = session.run(None, {"X": x, "max": np.array(1, dtype=np.float32)})[0]
        np.testing.assert_array_equal(result, np.minimum(x, 1))

    def test_node_proto_bool_constants(self):
        for shape in ([], [1]):
            with self.subTest(shape=shape):
                node = helper.make_node(
                    "Constant",
                    [],
                    ["Y"],
                    value=helper.make_tensor("value", TensorProto.BOOL, shape, [True]),
                )
                result = ExtendedReferenceEvaluator(node).run(None, {})[0]
                self.assertEqual(result.shape, tuple(shape))
                self.assertEqual(result.dtype, np.bool_)
                self.assertTrue(result.all())
                np.testing.assert_array_equal(
                    ExtendedReferenceEvaluator(node).run(None)[0], result
                )

    def test_node_proto_captured_subgraph_inputs(self):
        then_branch = helper.make_graph(
            [helper.make_node("Identity", ["X"], ["Y"])],
            "then",
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        )
        else_branch = helper.make_graph(
            [helper.make_node("Neg", ["X"], ["Y"])],
            "else",
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        )
        session = ExtendedReferenceEvaluator(
            helper.make_node(
                "If", ["condition"], ["result"], then_branch=then_branch, else_branch=else_branch
            )
        )
        self.assertEqual(session.input_names, ["condition", "X"])
        for condition in (True, False):
            with self.subTest(condition=condition):
                result = session.run(
                    None, {"condition": np.array(condition), "X": self.feeds["X"]}
                )[0]
                np.testing.assert_array_equal(
                    result, self.feeds["X"] if condition else -self.feeds["X"]
                )

    def test_explicit_native_custom_callback(self):
        session = ExtendedReferenceEvaluator(self.make_model())
        session.register_custom_kernel("", "Add", lambda node, x, b: x - b)
        result = session.run(["sum"], self.feeds)[0]
        np.testing.assert_array_equal(
            result, self.feeds["X"] - np.array([1, -1], dtype=np.float32)
        )
        self.assertTrue(session.unregister_custom_kernel("", "Add"))
        self.assertFalse(session.unregister_custom_kernel("", "Add"))
        np.testing.assert_array_equal(session.run(["sum"], self.feeds)[0], self.expected_sum)

    def test_greater_float64_broadcast_precision(self):
        x = np.array([[1 + 1e-12, 1.0], [1 - 1e-12, 1 + 3e-12]], dtype=np.float64)
        y = np.array([1.0, 1 + 2e-12], dtype=np.float64)
        session = ExtendedReferenceEvaluator(helper.make_node("Greater", ["X", "Y"], ["Z"]))
        result = session.run(None, {"X": x, "Y": y})[0]
        np.testing.assert_array_equal(
            result, np.array([[True, False], [False, True]], dtype=np.bool_)
        )
        self.assertEqual(result.dtype, np.bool_)
        self.assertEqual(x.dtype, np.float64)

    def test_arg_reduction_float64_precision_and_ties(self):
        x = np.array(
            [[1.0, 1 + 1e-12], [2 + 1e-12, 2.0], [-1.0, -1 + 1e-12], [3.0, 3.0]], dtype=np.float64
        )
        for op_type in ("ArgMax", "ArgMin"):
            for axis in (0, 1, -1):
                for keepdims in (0, 1):
                    for select_last_index in (0, 1):
                        with self.subTest(
                            op_type=op_type,
                            axis=axis,
                            keepdims=keepdims,
                            select_last_index=select_last_index,
                        ):
                            node = helper.make_node(
                                op_type,
                                ["X"],
                                ["Y"],
                                axis=axis,
                                keepdims=keepdims,
                                select_last_index=select_last_index,
                            )
                            result = ExtendedReferenceEvaluator(node).run(None, {"X": x})[0]
                            if axis == 0:
                                expected = [3, 3] if op_type == "ArgMax" else [2, 2]
                            else:
                                expected = (
                                    [1, 0, 1, select_last_index]
                                    if op_type == "ArgMax"
                                    else [0, 1, 0, select_last_index]
                                )
                            expected = np.array(expected, dtype=np.int64)
                            if keepdims:
                                expected = np.expand_dims(expected, axis=axis)
                            np.testing.assert_array_equal(result, expected)
                            self.assertEqual(result.dtype, np.int64)

    def test_arg_reduction_defaults_and_integer_precision(self):
        x = np.array([2**60, 2**60 + 1, 2**60 - 1], dtype=np.int64)
        for op_type, expected in (("ArgMax", 1), ("ArgMin", 2)):
            with self.subTest(op_type=op_type):
                session = ExtendedReferenceEvaluator(helper.make_node(op_type, ["X"], ["Y"]))
                result = session.run(None, {"X": x})[0]
                np.testing.assert_array_equal(result, np.array([expected], dtype=np.int64))
                scalar = ExtendedReferenceEvaluator(
                    helper.make_node(op_type, ["X"], ["Y"], keepdims=0)
                ).run(None, {"X": x})[0]
                self.assertEqual(scalar.shape, ())
                self.assertEqual(scalar.dtype, np.int64)
                self.assertEqual(scalar.item(), expected)

    def test_explicit_project_kernel(self):
        class Add(NativeOpKernel):
            def _run(self, x, b):
                return (x - b,)

        session = ExtendedReferenceEvaluator(self.make_model(), new_ops=[Add])
        result = session.run(["sum"], self.feeds)[0]
        np.testing.assert_array_equal(
            result, self.feeds["X"] - np.array([1, -1], dtype=np.float32)
        )

    def test_function_attributes(self):
        node = helper.make_node("LeakyRelu", ["X"], ["Y"])
        attribute = node.attribute.add()
        attribute.name = "alpha"
        attribute.ref_attr_name = "slope"
        attribute.type = 1
        function = helper.make_function(
            "test.native",
            "Activation",
            ["X"],
            ["Y"],
            [node],
            opset_imports=[helper.make_opsetid("", 18)],
            attributes=["slope"],
        )
        x = np.array([-2, 2], dtype=np.float32)
        result = ExtendedReferenceEvaluator(function).run(
            None, {"X": x}, attributes={"slope": 0.25}
        )
        np.testing.assert_array_equal(result[0], np.array([-0.5, 2], dtype=np.float32))

    def test_function_calls_an_external_function(self):
        inner = helper.make_function(
            "test.native",
            "Double",
            ["X"],
            ["Y"],
            [helper.make_node("Add", ["X", "X"], ["Y"])],
            opset_imports=[helper.make_opsetid("", 18)],
        )
        outer = helper.make_function(
            "test.native",
            "Twice",
            ["X"],
            ["Y"],
            [helper.make_node("Double", ["X"], ["Y"], domain="test.native")],
            opset_imports=[helper.make_opsetid("test.native", 1)],
        )
        result = ExtendedReferenceEvaluator(outer, functions=[inner]).run(
            None, {"X": self.feeds["X"]}
        )
        np.testing.assert_array_equal(result[0], self.feeds["X"] * 2)

    def test_invalid_kernel_and_input_count(self):
        with self.assertRaisesRegex(TypeError, "NativeOpKernel"):
            ExtendedReferenceEvaluator(self.make_model(), new_ops=[object])
        with self.assertRaisesRegex(ValueError, "Expected 2 inputs"):
            ExtendedReferenceEvaluator(self.make_model()).run([self.feeds["X"]])

    def test_missing_native_kernel_is_not_replaced(self):
        model = self.make_model()
        model.graph.node[0].op_type = "IntentionallyUnavailable"
        with self.assertRaisesRegex(ValueError, "unsupported op_type"):
            ExtendedReferenceEvaluator(model).run(None, self.feeds)


if __name__ == "__main__":
    unittest.main()
