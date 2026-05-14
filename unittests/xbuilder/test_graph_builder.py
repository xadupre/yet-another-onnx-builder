import unittest
from typing import Dict, List
import onnx.helper as oh
import numpy as np
import onnx.numpy_helper as onh
from onnx import AttributeProto, FunctionProto, GraphProto, TensorProto
from yobx.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    requires_onnxir,
    requires_torch,
    requires_onnxscript,
)
from yobx.reference import ExtendedReferenceEvaluator
from yobx.xbuilder import GraphBuilder, FunctionOptions, OptimizationOptions
from yobx.container import ExtendedModelContainer, ExportArtifact

TFLOAT = TensorProto.FLOAT
TFLOAT16 = TensorProto.FLOAT16
TINT64 = TensorProto.INT64


class TestGraphBuilder(ExtTestCase):
    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_inline_1_function(self):
        new_domain = "custom"

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [oh.make_node("MatMul", ["x", "a"], ["xa"]), oh.make_node("Add", ["xa", "b"], ["y"])],
            [oh.make_opsetid("", 14)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain),
                oh.make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
        )
        ref = ExtendedReferenceEvaluator(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(onnx_model)
        self.assertEqual(len(gr.functions), 1)
        onx = gr.to_onnx(inline=False)
        self.assertEqual(len(onx.functions), 1)

        self.assertRaise(
            lambda: gr.to_onnx(
                function_options=FunctionOptions(export_as_function=True, name="lr")
            ),
            AssertionError,
        )
        gr.inline_functions(verbose=1)
        function_proto = gr.to_onnx(
            function_options=FunctionOptions(
                export_as_function=True, name="lr", domain="custom_domain"
            ),
            inline=False,
        )
        self.assertNotEmpty(function_proto)

        onx = gr.to_onnx(inline=False)
        self.assertEqual(len(gr.functions), 0)
        self.assertEqual(len(onx.functions), 0)
        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_inline_2_functions(self):
        new_domain = "custom"

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [oh.make_node("MatMul", ["x", "a"], ["xa"]), oh.make_node("Add", ["xa", "b"], ["y"])],
            [oh.make_opsetid("", 14)],
            [],
        )

        linear_add = oh.make_function(
            new_domain,
            "LinearAdd",
            ["x", "a"],
            ["y"],
            [oh.make_node("Add", ["x", "a"], ["y"])],
            [oh.make_opsetid("", 14)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain),
                oh.make_node("LinearAdd", ["Y1", "B"], ["Y2"], domain=new_domain),
                oh.make_node("Abs", ["Y2"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression, linear_add],
        )
        ref = ExtendedReferenceEvaluator(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(onnx_model)
        self.assertEqual(len(gr.functions), 2)
        onx = gr.to_onnx(inline=False)
        self.assertEqual(len(onx.functions), 2)

        gr.inline_functions()
        function_proto = gr.to_onnx(
            function_options=FunctionOptions(name="lr", domain="custom_domain")
        )
        self.assertNotEmpty(function_proto)

        onx = gr.to_onnx()
        self.assertEqual(len(gr.functions), 0)
        self.assertEqual(len(onx.functions), 0)
        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_inline_2_functions_recursive(self):
        new_domain = "custom"

        linear_add = oh.make_function(
            new_domain,
            "LinearAdd",
            ["x", "a"],
            ["y"],
            [oh.make_node("Add", ["x", "a"], ["y"])],
            [oh.make_opsetid("", 14)],
            [],
        )

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("LinearAdd", ["xa", "b"], ["y"], domain=new_domain),
            ],
            [oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y2"], domain=new_domain),
                oh.make_node("Abs", ["Y2"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_add, linear_regression],
        )
        ref = ExtendedReferenceEvaluator(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(onnx_model)
        self.assertEqual(len(gr.functions), 2)
        onx = gr.to_onnx(inline=False)
        self.assertEqual(len(onx.functions), 2)

        gr.inline_functions()
        function_proto = gr.to_onnx(
            function_options=FunctionOptions(name="lr", domain="custom_domain"), inline=False
        )
        self.assertNotEmpty(function_proto)

        onx = gr.to_onnx(inline=False)
        self.assertEqual(len(gr.functions), 0)
        self.assertEqual(len(onx.functions), 0)
        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @ignore_warnings(DeprecationWarning)
    def test_as_function_constant_notfull(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        np_weights = np.random.randn(4, 3).astype(np.float32)
        np_bias = np.random.randn(1, 3).astype(np.float32)
        init = g.make_initializer("weights", np_weights)
        bias = g.make_initializer("bias", np_bias)
        g.op.Add(g.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        g.make_tensor_output("Y", indexed=False)
        g.move_initializers_to_constant(full_parameter_name=False)
        fct = g.to_onnx(function_options=FunctionOptions(name="linear", domain="mine"))
        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        expected = feeds["X"] @ np_weights + np_bias
        ref = ExtendedReferenceEvaluator(fct)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_as_function_constant_full(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        np_weights = np.random.randn(4, 3).astype(np.float32)
        np_bias = np.random.randn(1, 3).astype(np.float32)
        init = g.make_initializer("weights", np_weights)
        bias = g.make_initializer("bias", np_bias)
        g.op.Add(g.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        g.make_tensor_output("Y", indexed=False)
        g.move_initializers_to_constant(full_parameter_name=True)
        fct = g.to_onnx(function_options=FunctionOptions(name="linear", domain="mine"))
        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        expected = feeds["X"] @ np_weights + np_bias
        ref = ExtendedReferenceEvaluator(fct)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_as_function_second(self):
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", None, None, False)
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
        np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 1000

        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", indexed=False)
        self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        new_inits, _ = g.make_local_function(
            gf,
            function_options=FunctionOptions(
                name="Regression",
                domain="custom",
                move_initializer_to_constant=False,
                return_initializer=True,
            ),
        )
        self.assertEqual(new_inits, ["weights", "bias"])
        self.assertEqualArray(g.initializers_dict["weights"], np_weights)

        bias2 = g.make_initializer("bias2", np_bias2)
        g.op.Add(
            g.anyop.Regression("X", *new_inits, name="linear", domain="custom"),
            bias2,
            outputs=["Y"],
        )
        g.make_tensor_output("Y", indexed=False)
        nodes = [(node.domain, node.op_type, node.input, node.output) for node in g.nodes]
        self.assertEqual(
            nodes,
            [
                ("custom", "Regression", ["X", "weights", "bias"], ["_onx_regression_X"]),
                ("", "Add", ["_onx_regression_X", "bias2"], ["Y"]),
            ],
        )

        # finally, the conversion to onnx
        text = g.pretty_text()
        self.assertIn("_onx_regression_X, bias2", text)
        fct = g.to_onnx(
            function_options=FunctionOptions(
                name="linear", domain="mine", return_initializer=True
            ),
            inline=False,
        )

        self.assertIsInstance(fct, ExportArtifact)
        self.assertIsNotNone(fct.function)
        self.assertIsInstance(fct.proto, FunctionProto)
        self.assertIsInstance(fct.function.nested_functions, list)
        self.assertTrue(all(isinstance(p, FunctionProto) for p in fct.function.nested_functions))
        self.assertIsInstance(fct.function.initializers_name, list)
        self.assertEqual(fct.function.initializers_name, ["weights", "bias2", "bias"])
        self.assertIsInstance(fct.function.initializers_dict, dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct.function.initializers_dict.values())
        )
        self.assertEqual(len(fct.function.initializers_name), len(fct.function.initializers_dict))
        proto = fct.proto
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(proto.input, ["X", "weights", "bias2", "bias"])
        self.assertEqual(proto.domain, "mine")
        self.assertEqual(proto.name, "linear")
        f1 = fct.function.nested_functions[0]
        self.assertEqual(f1.domain, "custom")
        self.assertEqual(f1.name, "Regression")
        self.assertEqual(f1.output, ["Y"])
        self.assertEqual(f1.input, ["X", "weights", "bias"])

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        feeds.update(fct.function.initializers_dict)
        self.assertEqualArray(np_weights, feeds["weights"])
        self.assertEqualArray(np_bias, feeds["bias"])
        self.assertEqualArray(np_bias2, feeds["bias2"])
        self.assertEqual(set(feeds), {"X", "weights", "bias2", "bias"})
        expected = feeds["X"] @ np_weights + np_bias + np_bias2
        ref = ExtendedReferenceEvaluator(fct.proto, functions=fct.function.nested_functions)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_as_function_nested_unique(self):
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
        np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 100
        np_bias3 = np.arange(3).reshape((1, 3)).astype(np.float32) + 1000

        # first function
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", None, None, False)
        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", indexed=False)
        self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

        # second function calling the first one
        g2 = GraphBuilder(18, ir_version=9, as_function=True)
        g2.make_tensor_input("X", None, None, False)
        new_inits, _ = g2.make_local_function(
            gf,
            function_options=FunctionOptions(
                name="Regression",
                domain="custom",
                move_initializer_to_constant=False,
                return_initializer=True,
            ),
        )

        bias2 = g2.make_initializer("bias2", np_bias2)
        g2.op.Add(
            g2.anyop.Regression("X", *new_inits, name="addc", domain="custom"),
            bias2,
            outputs=["Y"],
        )
        g2.make_tensor_output("Y", indexed=False)

        # a last step
        # second function calling the first one
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        new_inits, _ = g.make_local_function(
            g2,
            function_options=FunctionOptions(
                name="RegressionBias",
                domain="custom",
                move_initializer_to_constant=False,
                return_initializer=True,
            ),
        )
        self.assertEqual(len(g.functions), 2)

        bias3 = g.make_initializer("bias3", np_bias3)
        g.op.Add(
            g.anyop.RegressionBias("X", *new_inits, name="add_d", domain="custom"),
            bias3,
            outputs=["Y"],
        )
        g.make_tensor_output("Y", indexed=False)

        # finally, the conversion to onnx
        self.assertIn("FUNC RegressionBias[custom]", g.pretty_text())

        fct = g.to_onnx(
            g2,
            function_options=FunctionOptions(
                name="linear", domain="mine", return_initializer=True
            ),
            inline=False,
        )

        self.assertIsInstance(fct, ExportArtifact)
        self.assertIsNotNone(fct.function)
        self.assertIsInstance(fct.proto, FunctionProto)
        self.assertIsInstance(fct.function.nested_functions, list)
        self.assertTrue(all(isinstance(p, FunctionProto) for p in fct.function.nested_functions))
        self.assertIsInstance(fct.function.initializers_name, list)
        self.assertEqual(fct.function.initializers_name, ["weights", "bias3", "bias2", "bias"])
        self.assertIsInstance(fct.function.initializers_dict, dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct.function.initializers_dict.values())
        )
        self.assertEqual(len(fct.function.initializers_name), len(fct.function.initializers_dict))
        proto = fct.proto
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(proto.input, ["X", "weights", "bias3", "bias2", "bias"])
        self.assertEqual(proto.domain, "mine")
        self.assertEqual(proto.name, "linear")
        self.assertEqual(2, len(fct.function.nested_functions))
        f1 = fct.function.nested_functions[0]
        self.assertEqual(f1.domain, "custom")
        self.assertEqual(f1.name, "Regression")
        self.assertEqual(f1.output, ["Y"])
        self.assertEqual(f1.input, ["X", "weights", "bias"])
        f2 = fct.function.nested_functions[1]
        self.assertEqual(f2.domain, "custom")
        self.assertEqual(f2.name, "RegressionBias")
        self.assertEqual(f2.output, ["Y"])
        self.assertEqual(f2.input, ["X", "weights", "bias2", "bias"])

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        feeds.update(fct.function.initializers_dict)
        self.assertEqualArray(np_weights, feeds["weights"])
        self.assertEqualArray(np_bias, feeds["bias"])
        self.assertEqualArray(np_bias2, feeds["bias2"])
        self.assertEqualArray(np_bias3, feeds["bias3"])
        self.assertEqual(set(feeds), {"X", "weights", "bias", "bias3", "bias2"})
        expected = feeds["X"] @ np_weights + np_bias + np_bias2 + np_bias3

        # Evaluation of a function
        self.assertIn("opset: '': 18", g.pretty_text())
        ref = ExtendedReferenceEvaluator(fct.proto, functions=fct.function.nested_functions)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

        # Same with a model
        proto = g.to_onnx(inline=False)
        self.assertEqual(len(proto.functions), 2)
        ref = ExtendedReferenceEvaluator(proto)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_as_function_second_twice(self):
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10

        # function 1
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", None, None, False)
        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", indexed=False)
        self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

        # main graph
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        new_inits, _ = g.make_local_function(
            gf,
            function_options=FunctionOptions(
                name="Regression",
                domain="custom",
                move_initializer_to_constant=False,
                return_initializer=True,
            ),
        )
        self.assertEqual(len(g.functions), 1)
        self.assertEqual(new_inits, ["weights", "bias"])
        self.assertEqualArray(g.initializers_dict["weights"], np_weights)

        # function 3: the same name but different
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", None, None, False)

        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Sub(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", indexed=False)
        self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

        self.assertEqual(len(g.functions), 1)
        new_inits_2, (domain_name, function_name) = g.make_local_function(
            gf,
            function_options=FunctionOptions(
                name="Regression",
                domain="custom",
                move_initializer_to_constant=False,
                return_initializer=True,
                rename_allowed=True,
            ),
        )
        self.assertEqual(len(g.functions), 2)
        self.assertEqual(new_inits, ["weights", "bias"])
        self.assertEqualArray(g.initializers_dict["weights"], np_weights)

        # two functions
        g.op.Add(
            g.anyop.Regression("X", *new_inits, name="linear", domain="custom"),
            g.make_node(function_name, ["X", *new_inits_2], name="linear", domain=domain_name),
            outputs=["Y"],
        )
        g.make_tensor_output("Y", indexed=False)
        self.assertEqual(len(g.functions), 2)

        # finally, the conversion to onnx
        fct = g.to_onnx(
            function_options=FunctionOptions(
                name="linear", domain="mine", return_initializer=True
            ),
            inline=False,
        )

        self.assertIsInstance(fct, ExportArtifact)
        self.assertIsNotNone(fct.function)
        self.assertIsInstance(fct.proto, FunctionProto)
        self.assertIsInstance(fct.function.nested_functions, list)
        self.assertTrue(all(isinstance(p, FunctionProto) for p in fct.function.nested_functions))
        self.assertIsInstance(fct.function.initializers_name, list)
        self.assertEqual(fct.function.initializers_name, ["weights", "bias"])
        self.assertIsInstance(fct.function.initializers_dict, dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct.function.initializers_dict.values())
        )
        self.assertEqual(len(fct.function.initializers_name), len(fct.function.initializers_dict))
        proto = fct.proto
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(proto.input, ["X", "weights", "bias"])
        self.assertEqual(proto.domain, "mine")
        self.assertEqual(proto.name, "linear")
        f1 = fct.function.nested_functions[0]
        self.assertEqual(f1.domain, "custom")
        self.assertEqual(f1.name, "Regression")
        self.assertEqual(f1.output, ["Y"])
        self.assertEqual(f1.input, ["X", "weights", "bias"])
        f2 = fct.function.nested_functions[1]
        self.assertEqual(f2.domain, "custom")
        self.assertEqual(f2.name, "Regression_l2l")
        self.assertEqual(f2.output, ["Y"])
        self.assertEqual(f2.input, ["X", "weights", "bias"])

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        feeds.update(fct.function.initializers_dict)
        expected = feeds["X"] @ np_weights + np_bias + feeds["X"] @ np_weights - np_bias
        ref = ExtendedReferenceEvaluator(fct.proto, functions=fct.function.nested_functions)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(DeprecationWarning)
    def test_as_function_nested_twice(self):

        def _make_function():
            np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
            np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
            np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 100

            # first function
            gf = GraphBuilder(18, ir_version=9, as_function=True)
            gf.make_tensor_input("X", None, None, False)
            init = gf.make_initializer("weights", np_weights)
            bias = gf.make_initializer("bias", np_bias)
            gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
            gf.make_tensor_output("Y", indexed=False)
            self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

            # second function calling the first one
            g2 = GraphBuilder(18, ir_version=9, as_function=True)
            g2.make_tensor_input("X", None, None, False)
            new_inits, _ = g2.make_local_function(
                builder=gf,
                function_options=FunctionOptions(
                    name="Regression",
                    domain="custom",
                    move_initializer_to_constant=False,
                    return_initializer=True,
                ),
            )

            bias2 = g2.make_initializer("bias2", np_bias2)
            g2.op.Add(
                g2.anyop.Regression("X", *new_inits, name="addc", domain="custom"),
                bias2,
                outputs=["Y"],
            )
            g2.make_tensor_output("Y", indexed=False)
            return g2

        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
        np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 100

        # a last step
        # second function calling the first one
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)

        # let's add the first function
        g1 = _make_function()
        new_inits_1, _ = g.make_local_function(
            g1,
            function_options=FunctionOptions(
                name="RegressionBias",
                domain="custom",
                move_initializer_to_constant=False,
                return_initializer=True,
            ),
        )
        self.assertEqual(len(g.functions), 2)
        # let's add the second function
        g2 = _make_function()
        new_inits_2, (domain_name, function_name) = g.make_local_function(
            g2,
            function_options=FunctionOptions(
                name="RegressionBias",
                domain="custom",
                move_initializer_to_constant=False,
                return_initializer=True,
                rename_allowed=True,
            ),
        )
        self.assertEqual(len(g.functions), 4)

        g.op.Add(
            g.anyop.RegressionBias("X", *new_inits_1, name="reg2", domain="custom"),
            g.make_node(function_name, ["X", *new_inits_2], name="reg2", domain=domain_name),
            outputs=["Y"],
        )
        g.make_tensor_output("Y", indexed=False)

        # finally, the conversion to onnx
        self.assertIn("FUNC RegressionBias[custom]", g.pretty_text())

        fct = g.to_onnx(
            function_options=FunctionOptions(
                name="linear", domain="mine", return_initializer=True
            ),
            inline=False,
        )

        self.assertIsInstance(fct, ExportArtifact)
        self.assertIsNotNone(fct.function)
        self.assertIsInstance(fct.proto, FunctionProto)
        self.assertIsInstance(fct.function.nested_functions, list)
        self.assertTrue(all(isinstance(p, FunctionProto) for p in fct.function.nested_functions))
        self.assertIsInstance(fct.function.initializers_name, list)
        self.assertEqual(fct.function.initializers_name, ["weights", "bias2", "bias"])
        self.assertIsInstance(fct.function.initializers_dict, dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct.function.initializers_dict.values())
        )
        self.assertEqual(len(fct.function.initializers_name), len(fct.function.initializers_dict))
        proto = fct.proto
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(proto.input, ["X", "weights", "bias2", "bias"])
        self.assertEqual(proto.domain, "mine")
        self.assertEqual(proto.name, "linear")
        self.assertEqual(4, len(fct.function.nested_functions))
        f1 = fct.function.nested_functions[0]
        self.assertEqual(f1.domain, "custom")
        self.assertEqual(f1.name, "Regression")
        self.assertEqual(f1.output, ["Y"])
        self.assertEqual(f1.input, ["X", "weights", "bias"])
        f2 = fct.function.nested_functions[1]
        self.assertEqual(f2.domain, "custom")
        self.assertEqual(f2.name, "RegressionBias")
        self.assertEqual(f2.output, ["Y"])
        self.assertEqual(f2.input, ["X", "weights", "bias2", "bias"])

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        feeds.update(fct.function.initializers_dict)
        self.assertEqualArray(np_weights, feeds["weights"])
        self.assertEqualArray(np_bias, feeds["bias"])
        self.assertEqualArray(np_bias2, feeds["bias2"])
        expected = (feeds["X"] @ np_weights + np_bias + np_bias2) * 2

        # Evaluation of a function
        self.assertIn("opset: '': 18", g.pretty_text())
        ref = ExtendedReferenceEvaluator(fct.proto, functions=fct.function.nested_functions)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

        # Same with a model
        proto = g.to_onnx(inline=False)
        self.assertEqual(len(proto.functions), 4)
        ref = ExtendedReferenceEvaluator(proto)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_as_function_nested_twice_merge(self):

        def _make_function():
            np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
            np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
            np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 100

            # first function
            gf = GraphBuilder(18, ir_version=9, as_function=True)
            gf.make_tensor_input("X", None, None, False)
            init = gf.make_initializer("weights", np_weights)
            bias = gf.make_initializer("bias", np_bias)
            gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
            gf.make_tensor_output("Y", indexed=False)
            self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

            # second function calling the first one
            g2 = GraphBuilder(18, ir_version=9, as_function=True)
            g2.make_tensor_input("X", None, None, False)
            new_inits, _ = g2.make_local_function(
                gf,
                function_options=FunctionOptions(
                    name="Regression",
                    domain="custom",
                    move_initializer_to_constant=False,
                    return_initializer=True,
                ),
            )

            bias2 = g2.make_initializer("bias2", np_bias2)
            g2.op.Add(
                g2.anyop.Regression("X", *new_inits, name="addc", domain="custom"),
                bias2,
                outputs=["Y"],
            )
            g2.make_tensor_output("Y", indexed=False)
            return g2

        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
        np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 100

        # a last step
        # second function calling the first one
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)

        # let's add the first function
        g1 = _make_function()
        new_inits_1, _ = g.make_local_function(
            g1,
            function_options=FunctionOptions(
                name="RegressionBias",
                domain="custom",
                move_initializer_to_constant=False,
                return_initializer=True,
            ),
        )
        self.assertEqual(len(g.functions), 2)
        # let's add the second function
        g2 = _make_function()
        new_inits_2, (domain_name, function_name) = g.make_local_function(
            g2,
            function_options=FunctionOptions(
                name="RegressionBias",
                domain="custom",
                move_initializer_to_constant=False,
                return_initializer=True,
                merge_allowed=True,
            ),
        )
        self.assertEqual(len(g.functions), 2)

        g.op.Add(
            g.anyop.RegressionBias("X", *new_inits_1, name="reg2", domain="custom"),
            g.make_node(function_name, ["X", *new_inits_2], name="reg2", domain=domain_name),
            outputs=["Y"],
        )
        g.make_tensor_output("Y", indexed=False)

        # finally, the conversion to onnx
        self.assertIn("FUNC RegressionBias[custom]", g.pretty_text())

        fct = g.to_onnx(
            function_options=FunctionOptions(
                name="linear", domain="mine", return_initializer=True
            ),
            inline=False,
        )

        self.assertIsInstance(fct, ExportArtifact)
        self.assertIsNotNone(fct.function)
        self.assertIsInstance(fct.proto, FunctionProto)
        self.assertIsInstance(fct.function.nested_functions, list)
        self.assertTrue(all(isinstance(p, FunctionProto) for p in fct.function.nested_functions))
        self.assertIsInstance(fct.function.initializers_name, list)
        self.assertEqual(fct.function.initializers_name, ["weights", "bias2", "bias"])
        self.assertIsInstance(fct.function.initializers_dict, dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct.function.initializers_dict.values())
        )
        self.assertEqual(len(fct.function.initializers_name), len(fct.function.initializers_dict))
        proto = fct.proto
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(proto.input, ["X", "weights", "bias2", "bias"])
        self.assertEqual(proto.domain, "mine")
        self.assertEqual(proto.name, "linear")
        self.assertEqual(2, len(fct.function.nested_functions))
        f1 = fct.function.nested_functions[0]
        self.assertEqual(f1.domain, "custom")
        self.assertEqual(f1.name, "Regression")
        self.assertEqual(f1.output, ["Y"])
        self.assertEqual(f1.input, ["X", "weights", "bias"])
        f2 = fct.function.nested_functions[1]
        self.assertEqual(f2.domain, "custom")
        self.assertEqual(f2.name, "RegressionBias")
        self.assertEqual(f2.output, ["Y"])
        self.assertEqual(f2.input, ["X", "weights", "bias2", "bias"])

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        feeds.update(fct.function.initializers_dict)
        self.assertEqualArray(np_weights, feeds["weights"])
        self.assertEqualArray(np_bias, feeds["bias"])
        self.assertEqualArray(np_bias2, feeds["bias2"])
        expected = (feeds["X"] @ np_weights + np_bias + np_bias2) * 2

        # Evaluation of a function
        self.assertIn("opset: '': 18", g.pretty_text())
        ref = ExtendedReferenceEvaluator(fct.proto, functions=fct.function.nested_functions)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

        # Same with a model
        proto = g.to_onnx(inline=False)
        self.assertEqual(len(proto.functions), 2)
        ref = ExtendedReferenceEvaluator(proto)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    @requires_onnxir("0.1.8")
    @requires_onnxscript()
    def test_large_model_onnxscript_ir(self):
        import onnx_ir as oir

        new_domain = "custom"

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [oh.make_node("MatMul", ["x", "a"], ["xa"]), oh.make_node("Add", ["xa", "b"], ["y"])],
            [oh.make_opsetid("", 14)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain),
                oh.make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["da", "db"])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            [
                onh.from_array(np.random.rand(1024, 1024).astype(np.float32), name="A"),
                onh.from_array(np.random.rand(1024).astype(np.float32), name="B"),
            ],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
        )
        ref = ExtendedReferenceEvaluator(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(onnx_model)
        self.assertEqual(len(gr.functions), 1)
        container = gr.to_onnx(inline=False, large_model=True)
        self.assertIsInstance(container, ExportArtifact)
        self.assertIsInstance(container.container, ExtendedModelContainer)
        filename = self.get_dump_file("test_large_model_onnxscript_ir.onnx")
        container.save(filename, True)
        ref2 = ExtendedReferenceEvaluator(filename)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

        # ir
        m = container.container.to_ir()
        proto = oir.to_proto(m)

        ref3 = ExtendedReferenceEvaluator(proto)
        got = ref3.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_set_type_shape_or_rank_with_shape_and_device(self):
        g = GraphBuilder(18)
        g.set_type("a", TFLOAT)
        g.set_shape("a", (2, 3))
        g.set_device("a", -1)

        g.set_type_shape_or_rank("b", "a")

        self.assertTrue(g.has_type("b"))
        self.assertEqual(g.get_type("b"), TFLOAT)
        self.assertTrue(g.has_shape("b"))
        self.assertEqual(g.get_shape("b"), (2, 3))
        self.assertTrue(g.has_device("b"))
        self.assertEqual(g.get_device("b"), -1)

    def test_set_type_shape_or_rank_with_rank_only(self):
        g = GraphBuilder(18)
        g.set_type("c", TINT64)
        g.set_rank("c", 3)

        g.set_type_shape_or_rank("d", "c")

        self.assertTrue(g.has_type("d"))
        self.assertEqual(g.get_type("d"), TINT64)
        self.assertTrue(g.has_rank("d"))
        self.assertEqual(g.get_rank("d"), 3)
        self.assertFalse(g.has_shape("d"))
        self.assertFalse(g.has_device("d"))

    def test_set_type_shape_or_rank_no_info(self):
        g = GraphBuilder(18)
        # When `like` has no type, shape, rank, or device, nothing should be set.
        g.set_type_shape_or_rank("e", "f")
        self.assertFalse(g.has_type("e"))
        self.assertFalse(g.has_shape("e"))
        self.assertFalse(g.has_rank("e"))
        self.assertFalse(g.has_device("e"))

    def test__apply_reshape_to_shape(self):
        g = GraphBuilder(18)
        cases = [
            (("batch", "cache+seq"), (-1,), ("batch*(cache+seq)",)),
            (("s44", 1, "s9"), (0, -1, 1), ("s44", "s9", 1)),
            ((44, 1, 9), (0, -1, 1), (44, 9, 1)),
            (("s23",), (-1, 1, 1, 1), ("s23", 1, 1, 1)),
            (("seq_length",), (1, 1, -1, 1), (1, 1, "seq_length", 1)),
            (("s31+seq_length",), (1, 1, 1, -1), (1, 1, 1, "s31+seq_length")),
            (
                ("s23", 1, "seq_length", "s31+seq_length"),
                (-1,),
                ("s23*(s31+seq_length)*seq_length",),
            ),
            (("s44", 16, 1), (0, 1, -1), ("s44", 1, 16)),
        ]
        for s1, s2, expected in cases:
            with self.subTest(case=(s1, s2, expected)):
                self.assertEqual(expected, g._apply_reshape_to_shape(s1, s2))

    def test_topological_order(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Equal", ["I", "B"], ["eq1"]),
                    oh.make_node("Not", ["eq1"], ["neq1"]),
                    oh.make_node("Where", ["neq1", "I", "zeroi"], ["ind"]),
                    oh.make_node("Unsqueeze", ["ind", "one"], ["flat_ind"]),
                    oh.make_node("LogSoftmax", ["X"], ["logX"], axis=1),
                    oh.make_node("GatherElements", ["logX", "flat_ind"], ["gx"], axis=1),
                    oh.make_node("Squeeze", ["gx", "one"], ["flat_gx"]),
                    oh.make_node("Neg", ["flat_gx"], ["neg_gx"]),
                    oh.make_node("Where", ["neq1", "neg_gx", "zerof"], ["w2"]),
                    oh.make_node("Cast", ["w2"], ["w2f"], to=TFLOAT),
                    oh.make_node("Cast", ["neq1"], ["neq1f"], to=TFLOAT),
                    oh.make_node(
                        "ReduceSum", ["w2f"], ["red1"], keepdims=0, noop_with_empty_axes=0
                    ),
                    oh.make_node(
                        "ReduceSum", ["neq1f"], ["red2"], keepdims=0, noop_with_empty_axes=0
                    ),
                    oh.make_node("Cast", ["red1"], ["red1_16"], to=TFLOAT16),
                    oh.make_node("Cast", ["red2"], ["red2_16"], to=TFLOAT16),
                    oh.make_node("Div", ["red1_16", "red2_16"], ["Y"]),
                ],
                "name",
                [
                    oh.make_tensor_value_info("X", TFLOAT16, ["A", "B"]),
                    oh.make_tensor_value_info("I", TINT64, ["A"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT16, [])],
                [
                    onh.from_array(np.array([-100], dtype=np.int64), name="B"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                    onh.from_array(np.array([0], dtype=np.float16), name="zerof"),
                    onh.from_array(np.array([0], dtype=np.int64), name="zeroi"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        feeds = dict(
            X=np.arange(12).reshape((3, 4)).astype(np.float16),
            I=np.array([2, 1, 0], dtype=np.int64),
        )
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(model)
        onx = gr.to_onnx()
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

        gr = GraphBuilder(model)
        gr.nodes = gr.nodes[::-1]
        gr.topological_sort()
        onx = gr.to_onnx()
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_inline_function_with_parameters(self):
        new_domain = "custom"

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["yeps"],
            [
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("Add", ["xa", "b"], ["y"]),
                oh.make_node("Constant", [], ["eps"]),
                oh.make_node("Add", ["y", "eps"], ["yeps"]),
            ],
            [oh.make_opsetid("", 14)],
            attributes=["epsilon"],
        )
        att = AttributeProto()
        att.name = "value_float"
        att.ref_attr_name = "epsilon"
        att.type = AttributeProto.FLOAT
        linear_regression.node[2].attribute.append(att)

        onnx_model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "LinearRegression",
                        ["X", "A", "B"],
                        ["Y1"],
                        domain=new_domain,
                        epsilon=10.0,
                    ),
                    oh.make_node("Abs", ["Y1"], ["Y"]),
                ],
                "example",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                    oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                    oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            ),
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
        )
        ref = ExtendedReferenceEvaluator(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(onnx_model, verbose=1)
        self.assertEqual(len(gr.functions), 1)
        onx = gr.to_onnx(inline=False)
        self.assertEqual(len(onx.functions), 1)

        self.assertRaise(
            lambda: gr.to_onnx(
                function_options=FunctionOptions(export_as_function=True, name="lr")
            ),
            AssertionError,
        )
        gr.inline_functions(verbose=1)
        function_proto = gr.to_onnx(
            function_options=FunctionOptions(
                export_as_function=True, name="lr", domain="custom_domain"
            ),
            inline=False,
        )
        self.assertNotEmpty(function_proto)

        onx = gr.to_onnx(inline=False)
        self.assertEqual(len(gr.functions), 0)
        self.assertEqual(len(onx.functions), 0)
        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def _get_cdist_implementation(
        self,
        node_inputs: List[str],
        node_outputs: List[str],
        opsets: Dict[str, int],
        domain="cdist_domain",
        metric="euclidean",
    ) -> FunctionProto:
        """Returns the CDist implementation as a function."""
        assert len(node_inputs) == 2, f"cdist has two inputs not {len(node_inputs)}."
        assert len(node_outputs) == 1, f"cdist has one outputs not {len(node_outputs)}."
        assert opsets, "opsets cannot be None."
        assert "" in opsets, f"Opsets for domain '' must be specified but opsets={opsets!r}."
        if opsets is not None and "com.microsoft" in opsets:
            node = oh.make_node(
                "CDist", ["xa", "xb"], ["z"], domain="com.microsoft", metric=metric
            )
            return oh.make_function(
                domain,
                f"CDist_{metric}",
                ["xa", "xb"],
                ["z"],
                [node],
                [oh.make_opsetid("com.microsoft", 1)],
            )

        if metric in ("euclidean", "sqeuclidean"):
            # subgraph
            nodes = [
                oh.make_node("Sub", ["next", "next_in"], ["diff"]),
                oh.make_node("Constant", [], ["axis"], value_ints=[1]),
                oh.make_node("ReduceSumSquare", ["diff", "axis"], ["scan_out"], keepdims=0),
                oh.make_node("Identity", ["next_in"], ["next_out"]),
            ]

            def make_value(name):
                value = oh.ValueInfoProto()
                value.name = name
                return value

            graph = oh.make_graph(
                nodes,
                "loop",
                [make_value("next_in"), make_value("next")],
                [make_value("next_out"), make_value("scan_out")],
            )

            scan = oh.make_node(
                "Scan", ["xb", "xa"], ["next_out", "zout"], num_scan_inputs=1, body=graph
            )
            final = (
                oh.make_node("Sqrt", ["zout"], ["z"])
                if metric == "euclidean"
                else oh.make_node("Identity", ["zout"], ["z"])
            )
            return oh.make_function(
                domain,
                f"CDist_{metric}",
                ["xa", "xb"],
                ["z"],
                [scan, final],
                [oh.make_opsetid("", opsets[""])],
            )

        raise RuntimeError(f"There is no implementation for cdist and metric={metric!r} yet.")

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_inline_function_with_subgraphs(self):
        def _make_model():
            new_domain = "custom"
            cdist = self._get_cdist_implementation(
                ["CX", "CY"], ["CZ"], domain="cdistdomain", opsets={"": 22}
            )

            bizarre = oh.make_function(
                new_domain,
                "BizarreRegression",
                ["x", "a", "b"],
                ["yfinal"],
                [
                    oh.make_node("MatMul", ["x", "a"], ["xa"]),
                    oh.make_node("Add", ["xa", "b"], ["y"]),
                    oh.make_node("Constant", [], ["eps"]),
                    oh.make_node("Add", ["y", "eps"], ["yeps"]),
                    oh.make_node(cdist.name, ["x", "yeps"], ["yfinal"], domain=cdist.domain),
                ],
                [oh.make_opsetid("", 22), oh.make_opsetid(cdist.domain, 1)],
                attributes=["epsilon"],
            )
            att = AttributeProto()
            att.name = "value_float"
            att.ref_attr_name = "epsilon"
            att.type = AttributeProto.FLOAT
            bizarre.node[2].attribute.append(att)

            onnx_model = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node(
                            bizarre.name,
                            ["X", "A", "B"],
                            ["Y1"],
                            domain=bizarre.domain,
                            epsilon=10.0,
                        ),
                        oh.make_node("Abs", ["Y1"], ["Y"]),
                    ],
                    "main_graph",
                    [
                        oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                        oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                        oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
                    ],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                ),
                opset_imports=[
                    oh.make_opsetid("", 22),
                    oh.make_opsetid(bizarre.domain, 1),
                    oh.make_opsetid(cdist.domain, 1),
                ],
                functions=[cdist, bizarre],
                ir_version=10,
            )
            return onnx_model

        onnx_model = _make_model()
        ref = self.check_ort(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(onnx_model, verbose=0)
        assert None not in gr.nodes
        self.assertEqual(len(gr.functions), 2)
        onx = gr.to_onnx(inline=False)
        self.assertNotIn(None, gr.nodes)
        self.dump_onnx("test_inline_function_with_subgraphs.onnx", onx)
        self.assertEqual(len(onx.functions), 2)
        gr = GraphBuilder(onnx_model, verbose=5)
        gr.inline_functions(verbose=1)
        function_proto = gr.to_onnx(
            function_options=FunctionOptions(
                export_as_function=True, name="lr", domain="custom_domain"
            ),
            inline=False,
        )
        self.assertNotEmpty(function_proto)

        onx = gr.to_onnx(inline=True)
        self.assertEqual(len(gr.functions), 0)
        self.assertEqual(len(onx.functions), 0)
        ref2 = self.check_ort(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def _get_cdist_implementation_with_ref_attribute(
        self,
        node_inputs: List[str],
        node_outputs: List[str],
        opsets: Dict[str, int],
        domain="cdist_domain",
        metric="euclidean",
    ) -> FunctionProto:
        """Returns the CDist implementation as a function."""
        assert len(node_inputs) == 2, f"cdist has two inputs not {len(node_inputs)}."
        assert len(node_outputs) == 1, f"cdist has one outputs not {len(node_outputs)}."
        assert opsets, "opsets cannot be None."
        assert "" in opsets, f"Opsets for domain '' must be specified but opsets={opsets!r}."
        assert opsets is not None and "com.microsoft" not in opsets
        if metric in ("euclidean", "sqeuclidean"):
            # subgraph
            nodes = [
                oh.make_node("Sub", ["next", "next_in"], ["diff"]),
                oh.make_node("Constant", [], ["axis"], value_ints=[1]),
                oh.make_node("Cast", ["diff"], ["diffc"]),
                oh.make_node("ReduceSumSquare", ["diffc", "axis"], ["out"], keepdims=0),
                oh.make_node("CastLike", ["out", "diff"], ["scan_out"]),
                oh.make_node("Identity", ["next_in"], ["next_out"]),
            ]
            att = AttributeProto()
            att.name = "to"
            att.ref_attr_name = "stash_type"
            att.type = AttributeProto.INT
            nodes[2].attribute.append(att)

            def make_value(name):
                value = oh.ValueInfoProto()
                value.name = name
                return value

            graph = oh.make_graph(
                nodes,
                "loop",
                [make_value("next_in"), make_value("next")],
                [make_value("next_out"), make_value("scan_out")],
            )

            scan = oh.make_node(
                "Scan", ["xb", "xa"], ["next_out", "zout"], num_scan_inputs=1, body=graph
            )
            final = (
                oh.make_node("Sqrt", ["zout"], ["z"])
                if metric == "euclidean"
                else oh.make_node("Identity", ["zout"], ["z"])
            )
            return oh.make_function(
                domain,
                f"CDist_{metric}",
                ["xa", "xb"],
                ["z"],
                [scan, final],
                [oh.make_opsetid("", opsets[""])],
                ["stash_type"],
            )

        raise RuntimeError(f"There is no implementation for cdist and metric={metric!r} yet.")

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_inline_function_with_subgraphs_with_ref_attribute(self):
        def _make_model():
            new_domain = "custom"
            cdist = self._get_cdist_implementation_with_ref_attribute(
                ["CX", "CY"], ["CZ"], domain="cdistdomain", opsets={"": 22}
            )

            bizarre = oh.make_function(
                new_domain,
                "BizarreRegression",
                ["x", "a", "b"],
                ["yfinal"],
                [
                    oh.make_node("MatMul", ["x", "a"], ["xa"]),
                    oh.make_node("Add", ["xa", "b"], ["y"]),
                    oh.make_node("Constant", [], ["eps"]),
                    oh.make_node("Add", ["y", "eps"], ["yeps"]),
                    oh.make_node(
                        cdist.name,
                        ["x", "yeps"],
                        ["yfinal"],
                        domain=cdist.domain,
                        stash_type=TensorProto.FLOAT,
                    ),
                ],
                [oh.make_opsetid("", 22), oh.make_opsetid(cdist.domain, 1)],
                attributes=["epsilon"],
            )
            att = AttributeProto()
            att.name = "value_float"
            att.ref_attr_name = "epsilon"
            att.type = AttributeProto.FLOAT
            bizarre.node[2].attribute.append(att)

            onnx_model = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node(
                            bizarre.name,
                            ["X", "A", "B"],
                            ["Y1"],
                            domain=bizarre.domain,
                            epsilon=10.0,
                        ),
                        oh.make_node("Abs", ["Y1"], ["Y"]),
                    ],
                    "main_graph",
                    [
                        oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                        oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                        oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
                    ],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                ),
                opset_imports=[
                    oh.make_opsetid("", 22),
                    oh.make_opsetid(bizarre.domain, 1),
                    oh.make_opsetid(cdist.domain, 1),
                ],
                functions=[cdist, bizarre],
                ir_version=10,
            )
            return onnx_model

        onnx_model = _make_model()
        ref = self.check_ort(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(onnx_model, verbose=0)
        assert None not in gr.nodes
        self.assertEqual(len(gr.functions), 2)
        onx = gr.to_onnx(inline=False)
        assert None not in gr.nodes
        self.assertEqual(len(onx.functions), 2)
        gr = GraphBuilder(onnx_model, verbose=5)
        gr.inline_functions(verbose=1)

        onx = gr.to_onnx(inline=False)
        self.dump_onnx("test_inline_function_with_subgraphs_with_ref_attribute.onnx", onx)
        self.assertEqual(len(gr.functions), 0)
        self.assertEqual(len(onx.functions), 0)
        ref2 = self.check_ort(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_inline_functions_subgraph(self):
        """Test that _inline_functions_subgraph inlines functions called inside a subgraph."""
        new_domain = "custom"

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [oh.make_node("MatMul", ["x", "a"], ["xa"]), oh.make_node("Add", ["xa", "b"], ["y"])],
            [oh.make_opsetid("", 18)],
            [],
        )

        # Build the then_branch: calls the custom function inside
        then_branch = oh.make_graph(
            [oh.make_node("LinearRegression", ["X", "A", "B"], ["Y_then"], domain=new_domain)],
            "then_branch",
            [],
            [oh.make_tensor_value_info("Y_then", TensorProto.FLOAT, None)],
        )

        # Build the else_branch: returns Abs(X)
        else_branch = oh.make_graph(
            [oh.make_node("Abs", ["X"], ["Y_else"])],
            "else_branch",
            [],
            [oh.make_tensor_value_info("Y_else", TensorProto.FLOAT, None)],
        )

        onnx_model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "If", ["Cond"], ["Y"], then_branch=then_branch, else_branch=else_branch
                    )
                ],
                "main_graph",
                [
                    oh.make_tensor_value_info("Cond", TensorProto.BOOL, []),
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                    oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                    oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
            ir_version=10,
        )

        feeds_true = dict(
            Cond=np.array(True),
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.eye(3).astype(np.float32),
            B=np.ones((3, 3), dtype=np.float32),
        )
        feeds_false = dict(
            Cond=np.array(False),
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.eye(3).astype(np.float32),
            B=np.ones((3, 3), dtype=np.float32),
        )

        ref = self.check_ort(onnx_model)
        expected_true = ref.run(None, feeds_true)[0]
        expected_false = ref.run(None, feeds_false)[0]

        gr = GraphBuilder(onnx_model, verbose=5)
        self.assertEqual(len(gr.functions), 1)
        # inline_functions triggers _inline_functions_subgraph on the If subgraphs
        gr.inline_functions(verbose=1)

        onx = gr.to_onnx(inline=False)
        self.dump_onnx("test_inline_functions_subgraph.onnx", onx)
        self.assertEqual(len(gr.functions), 0)
        self.assertEqual(len(onx.functions), 0)

        ref2 = self.check_ort(onx)
        got_true = ref2.run(None, feeds_true)[0]
        got_false = ref2.run(None, feeds_false)[0]
        self.assertEqualArray(expected_true, got_true)
        self.assertEqualArray(expected_false, got_false)

    @requires_torch("2.0")
    def test_register_dynamic_object_from_dynamic_shapes_dict_wrap_dim(self):
        # WrapDim (string) case: string is pre-processed to WrapDim
        g = GraphBuilder(18, ir_version=9, dynamic_shapes={"args_0": {0: "batch"}})
        self.assertIn("batch", g.dynamic_objects)
        self.assertIn("batch", g.dynamic_dimensions_source)
        self.assertEqual(
            g.dynamic_dimensions_source["batch"], [{"input_name": "args_0", "axis": 0}]
        )

    @requires_torch("2.0")
    def test_register_dynamic_object_from_dynamic_shapes_dict_dim(self):
        import torch

        # _Dim case: torch.export.Dim
        batch = torch.export.Dim("batch", min=1, max=128)
        g = GraphBuilder(18, ir_version=9, dynamic_shapes={"args_0": {0: batch}})
        self.assertIn("batch", g.dynamic_objects)
        self.assertIn("batch", g.dynamic_dimensions_source)
        self.assertEqual(
            g.dynamic_dimensions_source["batch"], [{"input_name": "args_0", "axis": 0}]
        )

    @requires_torch("2.0")
    def test_register_dynamic_object_from_dynamic_shapes_dict_derived_dim(self):
        import torch

        # _DerivedDim case: derived from a base Dim
        base = torch.export.Dim("base", min=1, max=64)
        derived = base * 2
        g = GraphBuilder(18, ir_version=9, dynamic_shapes={"args_0": {0: derived}})
        self.assertIn("2*base", g.dynamic_objects)
        self.assertIn("base", g.dynamic_objects)
        self.assertIn("2*base", g.dynamic_dimensions_source)
        self.assertEqual(
            g.dynamic_dimensions_source["2*base"], [{"input_name": "args_0", "axis": 0}]
        )

    @requires_torch("2.0")
    def test_register_dynamic_object_from_dynamic_shapes_dict_none(self):
        # None value: no dynamic object registered for that axis
        g = GraphBuilder(18, ir_version=9, dynamic_shapes={"args_0": {0: None}})
        self.assertEqual(g.dynamic_objects, {})
        self.assertEqual(g.dynamic_dimensions_source, {})

    @requires_torch("2.0")
    def test_register_dynamic_object_from_dynamic_shapes_dict_invalid_type(self):
        # Invalid type should raise AssertionError
        self.assertRaise(
            lambda: GraphBuilder(18, ir_version=9, dynamic_shapes={"args_0": {0: 123}}),
            AssertionError,
        )

    @requires_torch("2.0")
    def test_register_dynamic_object_from_dynamic_shapes_dict_multiple_axes(self):
        # Multiple axes for the same input
        g = GraphBuilder(18, ir_version=9, dynamic_shapes={"args_0": {0: "batch", 1: "seq"}})
        self.assertIn("batch", g.dynamic_objects)
        self.assertIn("seq", g.dynamic_objects)
        self.assertEqual(
            g.dynamic_dimensions_source["batch"], [{"input_name": "args_0", "axis": 0}]
        )
        self.assertEqual(
            g.dynamic_dimensions_source["seq"], [{"input_name": "args_0", "axis": 1}]
        )

    @requires_torch("2.0")
    def test_register_dynamic_object_from_dynamic_shapes_dict_multiple_inputs(self):
        # Multiple inputs with dynamic shapes
        g = GraphBuilder(
            18,
            ir_version=9,
            dynamic_shapes={"args_0": {0: "batch"}, "args_1": {0: "batch", 1: "seq"}},
        )
        self.assertIn("batch", g.dynamic_objects)
        self.assertIn("seq", g.dynamic_objects)
        # "batch" source should include both inputs
        sources = g.dynamic_dimensions_source["batch"]
        self.assertEqual(len(sources), 2)

    @requires_torch("2.0")
    def test_dynamic_to_str_dim(self):
        import torch

        batch = torch.export.Dim("batch", min=1, max=128)
        g = GraphBuilder(18, ir_version=9)
        result = g._dynamic_to_str(batch)
        self.assertEqual(result, "batch")

    @requires_torch("2.0")
    def test_dynamic_to_str_derived_dim(self):
        import torch

        base = torch.export.Dim("base", min=1, max=64)
        derived = base * 2
        g = GraphBuilder(18, ir_version=9)
        result = g._dynamic_to_str(derived)
        self.assertEqual(result, derived.__name__)

    def test_update_model_with_parameter_renaming(self):
        """Test _update_model_with_parameter_renaming renames initializer in nodes."""
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, (2, 4))
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32)
        w_init = g.make_initializer("p_layer_weight", np_weights, parameter_name="layer.weight")
        self.assertEqual(g._parameter_renaming, {"p_layer_weight": "layer.weight"})
        g.op.MatMul("X", w_init, outputs=["Y"])
        g.make_tensor_output("Y", TFLOAT, (2, 3), indexed=False)
        onx = g.to_onnx()
        # The initializer must carry the external parameter name.
        init_names = [i.name for i in onx.graph.initializer]
        self.assertIn("layer.weight", init_names)
        self.assertNotIn("p_layer_weight", init_names)
        # The MatMul node must reference the renamed initializer.
        matmul_inputs = list(onx.graph.node[0].input)
        self.assertIn("layer.weight", matmul_inputs)
        self.assertNotIn("p_layer_weight", matmul_inputs)
        # Verify numerical correctness.
        feeds = {"X": np.random.randn(2, 4).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(feeds["X"] @ np_weights, got)

    def test_update_model_with_parameter_renaming_multiple(self):
        """Test _update_model_with_parameter_renaming with multiple renamed parameters."""
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, (2, 4))
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32)
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10.0
        w_init = g.make_initializer("p_w", np_weights, parameter_name="fc.weight")
        b_init = g.make_initializer("p_b", np_bias, parameter_name="fc.bias")
        self.assertEqual(g._parameter_renaming, {"p_w": "fc.weight", "p_b": "fc.bias"})
        mm = g.op.MatMul("X", w_init, outputs=["mm"])
        g.op.Add(mm, b_init, outputs=["Y"])
        g.make_tensor_output("Y", TFLOAT, (2, 3), indexed=False)
        onx = g.to_onnx()
        init_names = [i.name for i in onx.graph.initializer]
        self.assertIn("fc.weight", init_names)
        self.assertIn("fc.bias", init_names)
        self.assertNotIn("p_w", init_names)
        self.assertNotIn("p_b", init_names)
        # Both nodes must reference the renamed initializers.
        all_node_inputs = [inp for node in onx.graph.node for inp in node.input]
        self.assertIn("fc.weight", all_node_inputs)
        self.assertIn("fc.bias", all_node_inputs)
        self.assertNotIn("p_w", all_node_inputs)
        self.assertNotIn("p_b", all_node_inputs)
        # Verify numerical correctness.
        feeds = {"X": np.random.randn(2, 4).astype(np.float32)}
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(feeds["X"] @ np_weights + np_bias, got)


@requires_torch()
class TestGetInputDynamicShape(ExtTestCase):
    def setUp(self):
        self.g = GraphBuilder(18, ir_version=9)

    def test_no_dynamic_shapes_returns_static_shape(self):
        shape = self.g.get_input_dynamic_shape("x", 0, (2, 3), dynamic_shapes=None)
        self.assertEqual(shape, (2, 3))

    def test_dynamic_shapes_tuple_info_none_returns_static_shape(self):
        shape = self.g.get_input_dynamic_shape("x", 0, (2, 3), dynamic_shapes=(None,))
        self.assertEqual(shape, (2, 3))

    def test_dynamic_shapes_tuple_info_dict_with_wrapdim(self):
        wrap = GraphBuilder.WrapDim("batch")
        shape = self.g.get_input_dynamic_shape("x", 0, (2, 3), dynamic_shapes=({0: wrap},))
        self.assertEqual(shape, ("batch", 3))

    def test_dynamic_shapes_dict_info_dict_with_wrapdim(self):
        wrap = GraphBuilder.WrapDim("batch")
        shape = self.g.get_input_dynamic_shape("x", 0, (2, 3), dynamic_shapes={"x": {0: wrap}})
        self.assertEqual(shape, ("batch", 3))

    def test_dynamic_shapes_tuple_info_list_with_named_dim(self):
        class FakeDim:
            __name__ = "seq"

        shape = self.g.get_input_dynamic_shape("x", 0, (2, 3), dynamic_shapes=([FakeDim, None],))
        self.assertEqual(shape, ("FakeDim", 3))

    def test_check_two_shapes_are_compatible_same_ints(self):
        g = GraphBuilder(18)
        # identical integer shapes: no exception
        g._check_two_shapes_are_compatible((2, 3), (2, 3), name="x")

    def test_check_two_shapes_are_compatible_rank_mismatch(self):
        g = GraphBuilder(18)
        self.assertRaises(
            AssertionError,
            lambda: g._check_two_shapes_are_compatible((2, 3), (2, 3, 4), name="x"),
        )

    def test_check_two_shapes_are_compatible_value_mismatch(self):
        g = GraphBuilder(18)
        self.assertRaises(
            AssertionError, lambda: g._check_two_shapes_are_compatible((2, 3), (2, 5), name="x")
        )

    def test_check_two_shapes_are_compatible_same_strings(self):
        g = GraphBuilder(18)
        # identical string dimensions: no exception
        g._check_two_shapes_are_compatible(("batch", "seq"), ("batch", "seq"), name="x")

    def test_check_two_shapes_are_compatible_different_strings(self):
        g = GraphBuilder(18)
        # different string dimensions: constraint registered, no exception
        g._check_two_shapes_are_compatible(("batch", "seq"), ("b", "s"), name="x")
        constraints = g.get_registered_constraints()
        self.assertIn("batch", constraints)
        self.assertIn("b", constraints["batch"])

    def test_check_two_shapes_are_compatible_mixed_int_string(self):
        g = GraphBuilder(18)
        # one int, one string: compatible (no exception)
        g._check_two_shapes_are_compatible((2, "seq"), (2, "seq"), name="x")
        g._check_two_shapes_are_compatible((2, "seq"), (2, "other"), name="x")

    def test_check_op_type_if_valid(self):
        g = GraphBuilder(18)
        then_graph = oh.make_graph(
            [oh.make_node("Add", ["x", "y"], ["result"])],
            "then_branch",
            [],
            [oh.make_tensor_value_info("result", TensorProto.FLOAT, None)],
        )
        else_graph = oh.make_graph(
            [oh.make_node("Sub", ["x", "y"], ["result"])],
            "else_branch",
            [],
            [oh.make_tensor_value_info("result", TensorProto.FLOAT, None)],
        )
        # Should not raise
        g._check_op_type(
            "If",
            ["cond"],
            ["result"],
            domain="",
            name="test_if",
            then_branch=then_graph,
            else_branch=else_graph,
        )

    def test_check_op_type_if_missing_branches(self):
        g = GraphBuilder(18)
        self.assertRaises(
            AssertionError,
            lambda: g._check_op_type("If", ["cond"], ["result"], domain="", name="test_if"),
        )

    def test_check_op_type_if_mismatched_outputs(self):
        g = GraphBuilder(18)
        then_graph = oh.make_graph(
            [],
            "then_branch",
            [],
            [
                oh.make_tensor_value_info("r1", TensorProto.FLOAT, None),
                oh.make_tensor_value_info("r2", TensorProto.FLOAT, None),
            ],
        )
        else_graph = oh.make_graph(
            [], "else_branch", [], [oh.make_tensor_value_info("result", TensorProto.FLOAT, None)]
        )
        self.assertRaises(
            AssertionError,
            lambda: g._check_op_type(
                "If",
                ["cond"],
                ["result"],
                domain="",
                name="test_if",
                then_branch=then_graph,
                else_branch=else_graph,
            ),
        )

    def test_check_op_type_scan(self):
        g = GraphBuilder(18)
        body = oh.make_graph(
            [
                oh.make_node("Add", ["sum_in", "next"], ["sum_out"]),
                oh.make_node("Identity", ["next"], ["scan_out"]),
            ],
            "scan_body",
            [
                oh.make_tensor_value_info("sum_in", TensorProto.FLOAT, None),
                oh.make_tensor_value_info("next", TensorProto.FLOAT, None),
            ],
            [
                oh.make_tensor_value_info("sum_out", TensorProto.FLOAT, None),
                oh.make_tensor_value_info("scan_out", TensorProto.FLOAT, None),
            ],
        )
        # Should not raise
        g._check_op_type(
            "Scan",
            ["initial", "scan_input"],
            ["final", "scan_output"],
            domain="",
            name="test_scan",
            body=body,
            num_scan_inputs=1,
        )

    def test_check_op_type_loop(self):
        g = GraphBuilder(18)
        body = oh.make_graph(
            [oh.make_node("Identity", ["v"], ["v_out"])],
            "loop_body",
            [
                oh.make_tensor_value_info("iter", TensorProto.INT64, []),
                oh.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
                oh.make_tensor_value_info("v", TensorProto.FLOAT, None),
            ],
            [
                oh.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
                oh.make_tensor_value_info("v_out", TensorProto.FLOAT, None),
            ],
        )
        # Should not raise
        g._check_op_type(
            "Loop", ["max_iter", "cond", "v"], ["v_final"], domain="", name="test_loop", body=body
        )

    @ignore_warnings(DeprecationWarning)
    def test_make_nodes(self):
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10

        # Sub-builder: X -> MatMul(X, weights) + bias -> Y
        sub = GraphBuilder(18, ir_version=9, as_function=True)
        sub.make_tensor_input("X", TFLOAT, (2, 4), False)
        init_w = sub.make_initializer("weights", np_weights)
        init_b = sub.make_initializer("bias", np_bias)
        sub.op.Add(sub.op.MatMul("X", init_w, name="mm"), init_b, name="add", outputs=["Y"])
        sub.make_tensor_output("Y", TFLOAT, (2, 3), indexed=False)

        # Main builder: uses make_nodes to incorporate the sub-builder's computation
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, (2, 4))
        result = g.make_nodes(sub, input_names=["X"], output_names=["output_0"], prefix="sub_")
        self.assertEqual(result, "output_0")
        g.make_tensor_output("output_0", TFLOAT, (2, 3))
        onx = g.to_onnx()

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        expected = feeds["X"] @ np_weights + np_bias
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_make_nodes_as_function(self):
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10

        # Sub-builder: X -> MatMul(X, weights) + bias -> Y
        sub = GraphBuilder(18, ir_version=9, as_function=True)
        sub.make_tensor_input("X", TFLOAT, (2, 4), False)
        init_w = sub.make_initializer("weights", np_weights)
        init_b = sub.make_initializer("bias", np_bias)
        sub.op.Add(sub.op.MatMul("X", init_w, name="mm"), init_b, name="add", outputs=["Y"])
        sub.make_tensor_output("Y", TFLOAT, (2, 3), indexed=False)

        # Main builder: uses make_nodes with function_options to export as a local function
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, (2, 4))
        result = g.make_nodes(
            sub,
            input_names=["X"],
            output_names=["output_0"],
            function_options=FunctionOptions(
                name="LinearRegression", domain="custom", move_initializer_to_constant=True
            ),
        )
        self.assertEqual(result, "output_0")
        g.make_tensor_output("output_0", TFLOAT, (2, 3))
        onx = g.to_onnx(inline=False)
        self.assertEqual(len(onx.functions), 1)

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        expected = feeds["X"] @ np_weights + np_bias
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    def test_move_node_position_can_move(self):
        # Node at position 2 (Relu) only uses graph input X,
        # so it can be moved to position 1 (before Neg which uses a).
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Abs", ["X"], ["a"]),  # pos 0: produces 'a'
                    oh.make_node("Neg", ["a"], ["b"]),  # pos 1: uses 'a' from pos 0
                    oh.make_node("Relu", ["X"], ["c"]),  # pos 2: uses 'X' (graph input)
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, [None])],
                [
                    oh.make_tensor_value_info("b", TFLOAT, [None]),
                    oh.make_tensor_value_info("c", TFLOAT, [None]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        gr = GraphBuilder(model)
        # Relu at pos 2: first_at.get('X', 0) == 0, can_be == 1 < 2 => can move
        new_pos = gr._move_node_position(2)
        self.assertEqual(new_pos, 1)
        self.assertEqual(gr.nodes[1].op_type, "Relu")
        self.assertEqual(gr.nodes[2].op_type, "Neg")

    def test_move_node_position_cannot_move(self):
        # Node at position 1 (Neg) uses 'a' produced at position 0,
        # so can_be == 1 == pos => cannot be moved, returns None.
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Abs", ["X"], ["a"]),  # pos 0: produces 'a'
                    oh.make_node("Neg", ["a"], ["b"]),  # pos 1: uses 'a' from pos 0
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, [None])],
                [oh.make_tensor_value_info("b", TFLOAT, [None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        gr = GraphBuilder(model)
        # Neg at pos 1: can_be == 0 + 1 == 1 >= 1 => cannot move
        new_pos = gr._move_node_position(1)
        self.assertIsNone(new_pos)
        # nodes order must remain unchanged
        self.assertEqual(gr.nodes[0].op_type, "Abs")
        self.assertEqual(gr.nodes[1].op_type, "Neg")

    def test_move_node_position_multiple_inputs(self):
        # Node at position 3 (Add) uses 'b' from pos 1 and 'c' from pos 2,
        # can_be == max(1, 2) + 1 == 3 == pos => cannot be moved.
        # Node at position 4 (Relu) only uses graph input X,
        # can_be == 0 + 1 == 1 < 4 => can be moved to position 1.
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Abs", ["X"], ["a"]),  # pos 0
                    oh.make_node("Neg", ["a"], ["b"]),  # pos 1
                    oh.make_node("Sigmoid", ["a"], ["c"]),  # pos 2
                    oh.make_node("Add", ["b", "c"], ["d"]),  # pos 3
                    oh.make_node("Relu", ["X"], ["e"]),  # pos 4: uses X only
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, [None])],
                [
                    oh.make_tensor_value_info("d", TFLOAT, [None]),
                    oh.make_tensor_value_info("e", TFLOAT, [None]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        gr = GraphBuilder(model)
        # Add at pos 3: can_be == max(1, 2) + 1 == 3 >= 3 => cannot move
        new_pos = gr._move_node_position(3)
        self.assertIsNone(new_pos)
        # A new GraphBuilder is needed to test position 4 independently,
        # since _move_node_position modifies self.nodes in-place.
        gr2 = GraphBuilder(model)
        new_pos2 = gr2._move_node_position(4)
        self.assertEqual(new_pos2, 1)
        self.assertEqual(gr2.nodes[1].op_type, "Relu")

    @ignore_warnings(DeprecationWarning)
    def test_rename_op_type_in_local_functions(self):
        # Build a FunctionProto with MatMul and Add nodes.
        func = oh.make_function(
            "custom",
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [oh.make_node("MatMul", ["x", "a"], ["xa"]), oh.make_node("Add", ["xa", "b"], ["y"])],
            [oh.make_opsetid("", 18)],
            [],
        )

        # Build a minimal model so GraphBuilder can be instantiated.
        graph = oh.make_graph(
            [oh.make_node("LinearRegression", ["X", "A", "B"], ["Y"], domain="custom")],
            "test",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )
        model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("custom", 1)],
            functions=[func],
            ir_version=9,
        )
        gr = GraphBuilder(model)

        # Case 1: no op in the proto matches the replacements → same object returned.
        result = gr._rename_op_type_in_local_functions(func, {("", "Relu"): ("", "Tanh")})
        self.assertIs(result, func)

        # Case 2: one op matches → a new proto is returned with the renamed op type.
        result = gr._rename_op_type_in_local_functions(func, {("", "Add"): ("", "Sub")})
        self.assertIsNot(result, func)
        self.assertIsInstance(result, FunctionProto)
        op_types = [(n.domain, n.op_type) for n in result.node]
        self.assertIn(("", "MatMul"), op_types)
        self.assertIn(("", "Sub"), op_types)
        self.assertNotIn(("", "Add"), op_types)

    @ignore_warnings(DeprecationWarning)
    def test_rename_op_type_in_local_functions_subgraph(self):
        # Build a FunctionProto whose body contains an If node whose subgraph
        # branches each contain an Add node that should be renamed to Sub.
        then_graph = oh.make_graph(
            [oh.make_node("Add", ["x", "b"], ["y"])],
            "then_branch",
            [],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        else_graph = oh.make_graph(
            [oh.make_node("Abs", ["x"], ["y"])],
            "else_branch",
            [],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        if_node = oh.make_node(
            "If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph
        )
        func = oh.make_function(
            "custom",
            "CondFunc",
            ["x", "b", "cond"],
            ["y"],
            [if_node],
            [oh.make_opsetid("", 18)],
            [],
        )

        graph = oh.make_graph(
            [oh.make_node("CondFunc", ["X", "B", "C"], ["Y"], domain="custom")],
            "test",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None]),
                oh.make_tensor_value_info("C", TensorProto.BOOL, []),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )
        model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("custom", 1)],
            functions=[func],
            ir_version=9,
        )
        gr = GraphBuilder(model)

        # The Add is inside a subgraph attribute (then_branch), not at the top level.
        result = gr._rename_op_type_in_local_functions(func, {("", "Add"): ("", "Sub")})
        self.assertIsNot(result, func)
        self.assertIsInstance(result, FunctionProto)

        # The If node itself should still be an If.
        self.assertEqual(len(result.node), 1)
        self.assertEqual(result.node[0].op_type, "If")

        # Collect the op types from the then-branch subgraph.
        then_att = next(a for a in result.node[0].attribute if a.name == "then_branch")
        self.assertIsInstance(then_att.g, GraphProto)
        then_ops = [(n.domain, n.op_type) for n in then_att.g.node]
        self.assertIn(("", "Sub"), then_ops)
        self.assertNotIn(("", "Add"), then_ops)

        # The else-branch should be unchanged (only Abs, no Add).
        else_att = next(a for a in result.node[0].attribute if a.name == "else_branch")
        else_ops = [(n.domain, n.op_type) for n in else_att.g.node]
        self.assertIn(("", "Abs"), else_ops)

    def test_optimize_node_subgraphs_inplace(self):
        # Build a model with an If node whose branches each contain an Identity
        # node.  After calling optimize_node_subgraphs_inplace the Identity node
        # should be eliminated from both branches.

        # then_branch: Add(x, x) -> tmp; Identity(tmp) -> y
        then_graph = oh.make_graph(
            [oh.make_node("Add", ["x", "x"], ["tmp"]), oh.make_node("Identity", ["tmp"], ["y"])],
            "then_branch",
            [],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        # else_branch: Abs(x) -> tmp; Identity(tmp) -> y
        else_graph = oh.make_graph(
            [oh.make_node("Abs", ["x"], ["tmp"]), oh.make_node("Identity", ["tmp"], ["y"])],
            "else_branch",
            [],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        if_node = oh.make_node(
            "If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph
        )
        graph = oh.make_graph(
            [if_node],
            "test",
            [
                oh.make_tensor_value_info("x", TensorProto.FLOAT, [None]),
                oh.make_tensor_value_info("cond", TensorProto.BOOL, []),
            ],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)
        gr = GraphBuilder(model, optimization_options=OptimizationOptions(recursive=True))
        node = gr.nodes[0]
        self.assertEqual(node.op_type, "If")

        # Both branches start with 2 nodes each.
        then_att = next(a for a in node.attribute if a.name == "then_branch")
        else_att = next(a for a in node.attribute if a.name == "else_branch")
        self.assertEqual(len(then_att.g.node), 2)
        self.assertEqual(len(else_att.g.node), 2)

        context = set(i.name for i in gr.inputs)
        gr.optimize_node_subgraphs_inplace(node, context)

        # After optimization the Identity nodes must have been removed.
        then_att = next(a for a in node.attribute if a.name == "then_branch")
        else_att = next(a for a in node.attribute if a.name == "else_branch")
        then_ops = [n.op_type for n in then_att.g.node]
        else_ops = [n.op_type for n in else_att.g.node]
        self.assertNotIn("Identity", then_ops)
        self.assertNotIn("Identity", else_ops)

    def test_set_sequence_and_get_sequence(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_sequence_input("seq", TFLOAT, None)
        self.assertTrue(g.is_sequence("seq"))
        info = g.get_sequence("seq")
        self.assertIn("dtype", info)
        self.assertEqual(info["dtype"], TFLOAT)

    def test_set_sequence_with_shapes(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_sequence_input("seq", TFLOAT, (3, 4))
        self.assertTrue(g.is_sequence("seq"))
        info = g.get_sequence("seq")
        self.assertEqual(info["dtype"], TFLOAT)
        self.assertIsNotNone(info["shapes"])
        self.assertEqual(info["ranks"], (2,))

    def test_is_sequence_false_for_non_sequence(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, (3, 4))
        self.assertFalse(g.is_sequence("X"))

    def test_set_sequence_update_existing(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_sequence_input("seq", TFLOAT, None)
        # calling set_sequence again with same dtype should not raise
        g.set_sequence("seq", TFLOAT, shapes=None, ranks=None)
        info = g.get_sequence("seq")
        self.assertEqual(info["dtype"], TFLOAT)

    @ignore_warnings(DeprecationWarning)
    def test_make_tensor_sequence_input_builds_valid_model(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_sequence_input("seq", TFLOAT, None)
        g.make_node("SequenceLength", ["seq"], ["length"], name="seq_len")
        g.make_tensor_output("length", TensorProto.INT64, shape=[], indexed=False)
        onx = g.to_onnx()

        # The graph input should be typed as a sequence
        inp = onx.graph.input[0]
        self.assertEqual(inp.name, "seq")
        self.assertTrue(inp.type.HasField("sequence_type"))
        self.assertEqual(inp.type.sequence_type.elem_type.tensor_type.elem_type, TFLOAT)

        # Running the model should return the number of tensors in the sequence
        tensors = [np.ones((3, 4), dtype=np.float32), np.zeros((3, 4), dtype=np.float32)]
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"seq": tensors})
        self.assertEqual(got[0], 2)

    def test_get_constant_as_shape_false(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TensorProto.FLOAT, (3, 4), False)
        value = np.array([2, 3, 4], dtype=np.int64)
        name = g.make_initializer("cst", value)
        result = g.get_constant(name, exc=True, as_shape=False)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqualArray(value, result)

    def test_get_constant_as_shape_true(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TensorProto.FLOAT, (3, 4), False)
        value = np.array([2, 3, 4], dtype=np.int64)
        name = g.make_initializer("cst", value)
        result = g.get_constant(name, exc=True, as_shape=True)
        self.assertEqual(result, (2, 3, 4))
        self.assertIsInstance(result, tuple)

    def test_get_constant_from_parent(self):
        parent = GraphBuilder(18, ir_version=9)
        parent.make_tensor_input("X", TensorProto.FLOAT, (3, 4), False)
        value = np.array([2, 3, 4], dtype=np.int64)
        cst_name = parent.make_initializer("cst", value)

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        result = child.get_constant_from_parent(cst_name, exc=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqualArray(value, result)

    def test_get_constant_from_parent_as_shape(self):
        parent = GraphBuilder(18, ir_version=9)
        parent.make_tensor_input("X", TensorProto.FLOAT, (3, 4), False)
        value = np.array([2, 3, 4], dtype=np.int64)
        cst_name = parent.make_initializer("cst", value)

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        result = child.get_constant_from_parent(cst_name, exc=True, as_shape=True)
        self.assertEqual(result, (2, 3, 4))
        self.assertIsInstance(result, tuple)

    def test_extract_input_names_from_args(self):
        gr = GraphBuilder(18)
        gr.make_tensor_input("X", TFLOAT, shape=("batch", "seq"))
        gr.make_tensor_input("Y", TFLOAT, shape=("batch", "seq"))
        gr.make_tensor_input("Z", TFLOAT, shape=("batch",))

        # plain string names that exist in the graph
        self.assertEqual(["X"], gr.extract_input_names_from_args(["X"]))
        self.assertEqual(["X", "Y"], gr.extract_input_names_from_args(["X", "Y"]))

        # unknown names are ignored
        self.assertEqual([], gr.extract_input_names_from_args(["unknown"]))
        self.assertEqual(["X"], gr.extract_input_names_from_args(["X", "unknown"]))

        # non-string values (e.g. int) are ignored
        self.assertEqual(["X"], gr.extract_input_names_from_args(["X", 42]))

        # nested list / tuple
        self.assertEqual(["X", "Y"], gr.extract_input_names_from_args([["X", "Y"]]))
        self.assertEqual(["X", "Y"], gr.extract_input_names_from_args([("X", "Y")]))

        # duplicates are removed while preserving order
        self.assertEqual(["X", "Y"], gr.extract_input_names_from_args(["X", "Y", "X"]))

        # slice: start/stop/step that are known names
        self.assertEqual(
            ["X", "Y", "Z"], gr.extract_input_names_from_args([slice("X", "Y", "Z")])
        )
        self.assertEqual(["X", "Y"], gr.extract_input_names_from_args([slice("X", "Y", None)]))

        # empty input
        self.assertEqual([], gr.extract_input_names_from_args([]))

    def test_make_shape_from_results_static(self):
        g = GraphBuilder(18, ir_version=9)
        result = g.make_shape_from_results([2, 3, 4])
        self.assertIsInstance(result, str)
        self.assertEqualArray(np.array([2, 3, 4], dtype=np.int64), g.initializers_dict[result])

    def test_make_shape_from_results_static_cached(self):
        g = GraphBuilder(18, ir_version=9)
        result1 = g.make_shape_from_results([2, 3, 4])
        result2 = g.make_shape_from_results([2, 3, 4])
        self.assertEqual(result1, result2)

    def test_make_shape_from_results_dynamic_scalar(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, ("batch", 3))
        shape_X = g.op.Shape("X", outputs=["shape_X"])
        self.assertEqual(shape_X, "shape_X")
        g.set_type("shape_X", TINT64)
        g.set_shape("shape_X", (2,))
        batch_dim = g.op.Gather("shape_X", np.array(0, dtype=np.int64), outputs=["batch_dim"])
        self.assertEqual(batch_dim, "batch_dim")
        self.assertEqual(batch_dim, "batch_dim")
        g.set_type("batch_dim", TINT64)
        g.set_shape("batch_dim", ())
        new_shape = g.make_shape_from_results(["batch_dim", 3])
        out = g.op.Reshape("X", new_shape, outputs=["Y"])
        self.assertEqual(out, "Y")
        g.set_type("Y", TFLOAT)
        g.set_shape("Y", ("batch", 3))
        g.make_tensor_output("Y", TFLOAT, ("batch", 3), indexed=False)
        onx = g.to_onnx(optimize=False)
        ref = self.check_ort(onx)
        x = np.arange(6).reshape(2, 3).astype(np.float32)
        got = ref.run(None, {"X": x})
        self.assertEqualArray(x, got[0])

    def test_make_shape_from_results_dynamic_scalar_optimize(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, ("batch", 3))
        shape_X = g.op.Shape("X", outputs=["shape_X"])
        self.assertEqual(shape_X, "shape_X")
        g.set_type("shape_X", TINT64)
        g.set_shape("shape_X", (2,))
        batch_dim = g.op.Gather("shape_X", np.array(0, dtype=np.int64), outputs=["batch_dim"])
        self.assertEqual(batch_dim, "batch_dim")
        self.assertEqual(batch_dim, "batch_dim")
        g.set_type("batch_dim", TINT64)
        g.set_shape("batch_dim", ())
        new_shape = g.make_shape_from_results(["batch_dim", 3])
        out = g.op.Reshape("X", new_shape, outputs=["Y"])
        self.assertEqual(out, "Y")
        g.set_type("Y", TFLOAT)
        g.set_shape("Y", ("batch", 3))
        g.make_tensor_output("Y", TFLOAT, ("batch", 3), indexed=False)
        onx = g.to_onnx(optimize=True)
        ref = self.check_ort(onx)
        x = np.arange(6).reshape(2, 3).astype(np.float32)
        got = ref.run(None, {"X": x})
        self.assertEqualArray(x, got[0])

    def test_make_shape_from_results_all_dynamic(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, ("batch", "seq", 3))
        shape_X = g.op.Shape("X", outputs=["shape_X"])
        self.assertEqual(shape_X, "shape_X")
        g.set_type("shape_X", TINT64)
        g.set_shape("shape_X", (3,))
        batch_dim = g.op.Gather("shape_X", np.array(0, dtype=np.int64), outputs=["batch_dim"])
        self.assertEqual(batch_dim, "batch_dim")
        g.set_type("batch_dim", TINT64)
        g.set_shape("batch_dim", ())
        seq_dim = g.op.Gather("shape_X", np.array(1, dtype=np.int64), outputs=["seq_dim"])
        self.assertEqual(seq_dim, "seq_dim")
        g.set_type("seq_dim", TINT64)
        g.set_shape("seq_dim", ())
        new_shape = g.make_shape_from_results(["batch_dim", "seq_dim", 3])
        out = g.op.Reshape("X", new_shape, outputs=["Y"])
        self.assertEqual(out, "Y")
        g.set_type("Y", TFLOAT)
        g.set_shape("Y", ("batch", "seq", 3))
        g.make_tensor_output("Y", TFLOAT, ("batch", "seq", 3), indexed=False)
        onx = g.to_onnx(optimize=False)
        ref = self.check_ort(onx)
        x = np.arange(12).reshape(2, 2, 3).astype(np.float32)
        got = ref.run(None, {"X": x})
        self.assertEqualArray(x, got[0])

    def test_evaluate_dimension_equality_with_constraints(self):
        g = GraphBuilder(18)

        # integer: dimension == d1 + d2
        self.assertTrue(g.evaluate_dimension_equality_with_constraints(5, 2, "+", 3))

        # integer: dimension != d1 + d2
        self.assertFalse(g.evaluate_dimension_equality_with_constraints(5, 2, "+", 4))

        # string: dimension exactly matches f"{d1}+{d2}"
        self.assertTrue(g.evaluate_dimension_equality_with_constraints("a+b", "a", "+", "b"))

        # string: dimension matches via registered constraint
        g.add_to_constraints("batch", "seq1+seq2")
        self.assertTrue(
            g.evaluate_dimension_equality_with_constraints("batch", "seq1", "+", "seq2")
        )

        # string: dimension proved equal via simplify_two_expressions (commutativity)
        self.assertTrue(g.evaluate_dimension_equality_with_constraints("a+b", "b", "+", "a"))

    def _make_node_with_attrs(self, **attrs):
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        for name, value in attrs.items():
            node.attribute.append(oh.make_attribute(name, value))
        return node

    @requires_torch()
    def test_get_attribute_with_default_int(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(axis=2)
        self.assertEqual(gr.get_attribute_with_default(node, "axis", 0), 2)

    @requires_torch()
    def test_get_attribute_with_default_ints(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(perm=[0, 2, 1])
        self.assertEqual(gr.get_attribute_with_default(node, "perm", []), [0, 2, 1])

    @requires_torch()
    def test_get_attribute_with_default_float(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(alpha=0.5)
        self.assertAlmostEqual(gr.get_attribute_with_default(node, "alpha", 1.0), 0.5)

    @requires_torch()
    def test_get_attribute_with_default_floats(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(scales=[1.0, 2.0, 3.0])
        self.assertEqual(gr.get_attribute_with_default(node, "scales", []), [1.0, 2.0, 3.0])

    @requires_torch()
    def test_get_attribute_with_default_string(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(mode=b"constant")
        self.assertEqual(gr.get_attribute_with_default(node, "mode", b""), b"constant")

    @requires_torch()
    def test_get_attribute_with_default_strings(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(keys=[b"hello", b"world"])
        self.assertEqual(gr.get_attribute_with_default(node, "keys", []), [b"hello", b"world"])

    @requires_torch()
    def test_get_attribute_with_default_missing(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(axis=1)
        self.assertEqual(gr.get_attribute_with_default(node, "missing", 42), 42)

    @requires_torch()
    def test_get_attribute_with_default_unsupported_type(self):
        import onnx

        gr = GraphBuilder(18, ir_version=9)
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        tensor = onh.from_array(np.array([1.0], dtype=np.float32))
        att = onnx.AttributeProto()
        att.name = "value"
        att.type = onnx.AttributeProto.TENSOR
        att.t.CopyFrom(tensor)
        node.attribute.append(att)
        self.assertRaise(lambda: gr.get_attribute_with_default(node, "value", None), TypeError)

    @requires_torch()
    def test_get_attributes_with_default_int(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(axis=3)
        self.assertEqual(gr.get_attributes_with_default(node, axis=0), {"axis": 3})

    @requires_torch()
    def test_get_attributes_with_default_ints(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(perm=[1, 0, 2])
        self.assertEqual(gr.get_attributes_with_default(node, perm=[]), {"perm": [1, 0, 2]})

    @requires_torch()
    def test_get_attributes_with_default_float(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(alpha=0.25)
        result = gr.get_attributes_with_default(node, alpha=1.0)
        self.assertAlmostEqual(result["alpha"], 0.25)

    @requires_torch()
    def test_get_attributes_with_default_floats(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(scales=[1.5, 2.5])
        self.assertEqual(gr.get_attributes_with_default(node, scales=[]), {"scales": [1.5, 2.5]})

    @requires_torch()
    def test_get_attributes_with_default_string(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(mode=b"nearest")
        self.assertEqual(gr.get_attributes_with_default(node, mode=b""), {"mode": b"nearest"})

    @requires_torch()
    def test_get_attributes_with_default_strings(self):
        gr = GraphBuilder(18, ir_version=9)
        node = self._make_node_with_attrs(keys=[b"a", b"b", b"c"])
        self.assertEqual(
            gr.get_attributes_with_default(node, keys=[]), {"keys": [b"a", b"b", b"c"]}
        )

    @requires_torch()
    def test_get_attributes_with_default_uses_default(self):
        gr = GraphBuilder(18, ir_version=9)
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        self.assertEqual(gr.get_attributes_with_default(node, axis=5), {"axis": 5})

    @requires_torch()
    def test_get_attributes_with_default_none_default_excluded(self):
        gr = GraphBuilder(18, ir_version=9)
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        self.assertEqual(gr.get_attributes_with_default(node, axis=None), {})

    @requires_torch()
    def test_get_attributes_with_default_unsupported_type(self):
        import onnx

        gr = GraphBuilder(18, ir_version=9)
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        tensor = onh.from_array(np.array([1.0], dtype=np.float32))
        att = onnx.AttributeProto()
        att.name = "value"
        att.type = onnx.AttributeProto.TENSOR
        att.t.CopyFrom(tensor)
        node.attribute.append(att)
        self.assertRaise(lambda: gr.get_attributes_with_default(node, value=None), TypeError)


class TestGraphBuilderGetTypeKnown(ExtTestCase):
    @requires_torch()
    def test_get_type_known_missing(self):
        gr = GraphBuilder(18, ir_version=9)
        self.assertIsNone(gr.get_type_known("unknown"))

    @requires_torch()
    def test_get_type_known_valid(self):
        import torch

        gr = GraphBuilder(18, ir_version=9)
        # Store a value with the expected structure:
        # (where, (prefix, (name, dtype, shape)))
        gr.set_shapes_types("x", "run_node", ("", ("x", torch.float16, torch.Size([2, 3]))))
        result = gr.get_type_known("x")
        self.assertEqual(result, TensorProto.FLOAT16)

    @requires_torch()
    def test_get_type_known_valid_float32(self):
        import torch

        gr = GraphBuilder(18, ir_version=9)
        gr.set_shapes_types("y", "run_node", ("", ("y", torch.float32, torch.Size([4]))))
        result = gr.get_type_known("y")
        self.assertEqual(result, TensorProto.FLOAT)

    @requires_torch()
    def test_get_type_known_invalid_no_exc(self):
        gr = GraphBuilder(18, ir_version=9)
        # Store a value with a structure that does not match the expected tuple pattern
        gr.set_shapes_types("z", "run_node", "not_a_tuple")
        result = gr.get_type_known("z", exc=False)
        self.assertIsNone(result)

    @requires_torch()
    def test_get_type_known_invalid_with_exc(self):
        gr = GraphBuilder(18, ir_version=9)
        # Store a value with a structure that does not match; exc=True should raise
        gr.set_shapes_types("w", "run_node", "not_a_tuple")
        self.assertRaises(AssertionError, lambda: gr.get_type_known("w", exc=True))

    def test_has_exact_same_constant_in_context_same(self):
        # Child and parent have identical small constants: should return True.
        parent = GraphBuilder(18, ir_version=9)
        np_cst = np.arange(4).reshape((2, 2)).astype(np.float32)
        parent.make_initializer("cst", np_cst)

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        child.make_initializer("cst", np_cst)

        result = child.has_exact_same_constant_in_context("cst")
        self.assertTrue(result)

    def test_has_exact_same_constant_in_context_different_values(self):
        # Same shape/type but different values: should return False.
        parent = GraphBuilder(18, ir_version=9)
        np_cst1 = np.arange(4).reshape((2, 2)).astype(np.float32)
        parent.make_initializer("cst", np_cst1)

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        np_cst2 = np_cst1 + 1.0
        child.make_initializer("cst", np_cst2)

        result = child.has_exact_same_constant_in_context("cst")
        self.assertFalse(result)

    def test_has_exact_same_constant_in_context_different_shape(self):
        # Same name but different shapes: should return False.
        parent = GraphBuilder(18, ir_version=9)
        parent.make_initializer("cst", np.arange(6).reshape((2, 3)).astype(np.float32))

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        child.make_initializer("cst", np.arange(4).reshape((2, 2)).astype(np.float32))

        result = child.has_exact_same_constant_in_context("cst")
        self.assertFalse(result)

    def test_has_exact_same_constant_in_context_different_type(self):
        # Same name and shape but different dtypes: should return False.
        parent = GraphBuilder(18, ir_version=9)
        np_cst = np.arange(4).reshape((2, 2)).astype(np.float32)
        parent.make_initializer("cst", np_cst)

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        child.make_initializer("cst", np_cst.astype(np.float64))

        result = child.has_exact_same_constant_in_context("cst")
        self.assertFalse(result)

    def test_has_exact_same_constant_in_context_large(self):
        # Constants with >= 128 elements: comparison is skipped, should return None.
        parent = GraphBuilder(18, ir_version=9)
        np_cst = np.arange(128).reshape((16, 8)).astype(np.float32)
        parent.make_initializer("cst", np_cst)

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        child.make_initializer("cst", np_cst)

        result = child.has_exact_same_constant_in_context("cst")
        self.assertIsNone(result)

    def test_has_exact_same_constant_in_context_not_in_child(self):
        # Name is only a constant in the parent, not in the child: should return False.
        parent = GraphBuilder(18, ir_version=9)
        parent.make_initializer("cst", np.arange(4).reshape((2, 2)).astype(np.float32))

        child = GraphBuilder(18, ir_version=9, _parent=parent)

        result = child.has_exact_same_constant_in_context("cst")
        self.assertFalse(result)

    def test_has_exact_same_constant_in_context_not_in_parent(self):
        # Name is a constant in the child but not in the parent: should return False.
        parent = GraphBuilder(18, ir_version=9)

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        child.make_initializer("cst", np.arange(4).reshape((2, 2)).astype(np.float32))

        result = child.has_exact_same_constant_in_context("cst")
        self.assertFalse(result)

    def test_make_subset_builder(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", TFLOAT, (2, 4), False)
        g.make_tensor_input("W", TFLOAT, (4, 3), False)

        sub = g.make_subset_builder(["X", "W"], name="LinearSub", domain="mydom")
        self.assertEqual(sub.input_names, ["X", "W"])
        sub.op.MatMul("X", "W", outputs=["Y"])
        sub.make_tensor_output("Y", indexed=False)

        fct = sub.to_onnx(function_options=FunctionOptions(name="LinearSub", domain="mydom"))
        self.assertIsInstance(fct, ExportArtifact)
        self.assertIsInstance(fct.proto, FunctionProto)
        self.assertEqual(list(fct.proto.input), ["X", "W"])
        self.assertEqual(list(fct.proto.output), ["Y"])
        self.assertEqual(fct.proto.domain, "mydom")
        self.assertEqual(fct.proto.name, "LinearSub")

        feeds = dict(
            X=np.arange(8).reshape((2, 4)).astype(np.float32),
            W=np.arange(12).reshape((4, 3)).astype(np.float32),
        )
        expected = feeds["X"] @ feeds["W"]
        ref = ExtendedReferenceEvaluator(fct)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_make_subset_builder_add_local_functions(self):
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", TFLOAT, None, False)
        gf.op.Relu("X", outputs=["Y"])
        gf.make_tensor_output("Y", indexed=False)

        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("A", TFLOAT, (2, 4), False)
        g.make_local_function(gf, function_options=FunctionOptions(name="MyRelu", domain="test"))
        g.anyop.MyRelu("A", outputs=["B"], domain="test")
        g.make_tensor_output("B", indexed=False)
        self.assertEqual(len(g.functions), 1)

        sub = g.make_subset_builder(
            ["A"], name="SubFunc", domain="subdom", add_local_functions=True
        )
        self.assertEqual(sub.input_names, ["A"])
        self.assertEqual(len(sub.functions), 1)
        sub.anyop.MyRelu("A", outputs=["C"], domain="test")
        sub.make_tensor_output("C", indexed=False)

        fct = sub.to_onnx(
            function_options=FunctionOptions(name="SubFunc", domain="subdom"), inline=False
        )
        self.assertIsInstance(fct, ExportArtifact)
        self.assertIsInstance(fct.proto, FunctionProto)
        self.assertEqual(list(fct.proto.input), ["A"])
        self.assertEqual(list(fct.proto.output), ["C"])
        self.assertEqual(fct.proto.domain, "subdom")
        self.assertEqual(fct.proto.name, "SubFunc")

    def test_same_shape_static(self):
        g = GraphBuilder(18)
        g._known_shapes["X"] = (3, 4)
        g._known_shapes["Y"] = (3, 4)
        self.assertTrue(g.same_shape("X", "Y"))

    def test_same_shape_static_different(self):
        g = GraphBuilder(18)
        g._known_shapes["X"] = (3, 4)
        g._known_shapes["Y"] = (3, 5)
        self.assertFalse(g.same_shape("X", "Y"))

    def test_same_shape_different_rank(self):
        g = GraphBuilder(18)
        g._known_shapes["X"] = (3, 4)
        g._known_shapes["Y"] = (3, 4, 5)
        self.assertFalse(g.same_shape("X", "Y"))

    def test_same_shape_dynamic_same_dim(self):
        g = GraphBuilder(18)
        g._known_shapes["X"] = ("batch", 4)
        g._known_shapes["Y"] = ("batch", 4)
        self.assertTrue(g.same_shape("X", "Y"))

    def test_same_shape_dynamic_linked_by_constraints(self):
        g = GraphBuilder(18)
        g._known_shapes["X"] = ("a", 4)
        g._known_shapes["Y"] = ("b", 4)
        g.add_to_constraints("a", "b")
        g.add_to_constraints("b", "a")
        self.assertTrue(g.same_shape("X", "Y"))

    def test_same_shape_dynamic_no_constraints(self):
        g = GraphBuilder(18)
        g._known_shapes["X"] = ("a", 4)
        g._known_shapes["Y"] = ("b", 4)
        self.assertFalse(g.same_shape("X", "Y"))

    def test_set_value_shape_constraint_dim_registration(self):
        # When a name already has a symbolic (string) value shape like ("batch",)
        # and set_value_shape is called with a concrete (int,) tuple,
        # the constraint should be registered for the symbolic dim name ("batch"),
        # not for the literal string "existing".
        g = GraphBuilder(18)
        g.make_tensor_input("X", TFLOAT, ("batch",))
        g._known_value_shape["batch_value"] = ("batch",)
        g._known_ranks["batch_value"] = 1
        g.set_value_shape("batch_value", (5,))
        self.assertIn("batch", g.constraints_)
        self.assertIn(5, g.constraints_["batch"])
        self.assertNotIn("existing", g.constraints_)

    def test_get_dimension_as_result_already_known(self):
        gr = GraphBuilder(18)
        gr.make_tensor_input("X", TFLOAT, ("batch", "seq"))
        self.assertTrue(gr.has_name("X"))
        # When the name is already a known result, return it unchanged.
        result = gr.get_dimension_as_result("X")
        self.assertEqual(result, "X")
        # No Shape/Gather nodes should have been created.
        self.assertEqual(len(gr.nodes), 0)

    def test_get_dimension_as_result_from_source(self):
        gr = GraphBuilder(18)
        gr.make_tensor_input("X", TFLOAT, ("batch", "seq"))
        # Manually register a source for the dynamic dimension.
        gr.dynamic_dimensions_source["batch"] = [{"input_name": "X", "axis": 0}]
        self.assertFalse(gr.has_name("batch"))
        result = gr.get_dimension_as_result("batch")
        self.assertEqual(result, "batch")
        # A Shape node and a Gather node should have been added.
        op_types = [n.op_type for n in gr.nodes]
        self.assertIn("Shape", op_types)
        self.assertIn("Gather", op_types)
        shape_node = next(n for n in gr.nodes if n.op_type == "Shape")
        self.assertEqual(list(shape_node.input), ["X"])
        gather_node = next(n for n in gr.nodes if n.op_type == "Gather")
        self.assertEqual(list(gather_node.output), ["batch"])

    def test_get_dimension_as_result_no_source_raises(self):
        gr = GraphBuilder(18)
        gr.make_tensor_input("X", TFLOAT, ("batch", "seq"))
        # No source registered for "batch" -> AssertionError.
        self.assertRaises(AssertionError, gr.get_dimension_as_result, "batch")

    def test_constant_is_equal_to(self):
        g = GraphBuilder(18, ir_version=9)

        # equal arrays
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        g.make_initializer("w", arr)
        self.assertTrue(g.constant_is_equal_to("w", arr.copy()))

        # different values
        arr2 = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        self.assertFalse(g.constant_is_equal_to("w", arr2))

        # dtype mismatch
        arr3 = arr.astype(np.float64)
        self.assertFalse(g.constant_is_equal_to("w", arr3))

        # shape mismatch
        arr4 = arr.reshape((3, 1))
        self.assertFalse(g.constant_is_equal_to("w", arr4))

        # empty array (shape (0,))
        empty = np.array([], dtype=np.float32)
        g.make_initializer("empty", empty, allow_empty=True)
        self.assertTrue(g.constant_is_equal_to("empty", empty.copy()))

        # scalar array
        scalar = np.array(5.0, dtype=np.float32)
        g.make_initializer("scalar", scalar)
        self.assertTrue(g.constant_is_equal_to("scalar", np.array(5.0, dtype=np.float32)))
        self.assertFalse(g.constant_is_equal_to("scalar", np.array(6.0, dtype=np.float32)))

        # TensorProto value
        arr_tp = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        tp = onh.from_array(arr_tp, name="w_tp")
        g.make_initializer("w_tp", arr_tp)
        self.assertTrue(g.constant_is_equal_to("w_tp", tp))

        # large array (size >= 30) always returns True regardless of values
        large = np.arange(30, dtype=np.float32)
        g.make_initializer("large", large)
        large_same = large.copy()
        self.assertTrue(g.constant_is_equal_to("large", large_same))
        large_different = np.zeros(30, dtype=np.float32)
        self.assertTrue(g.constant_is_equal_to("large", large_different))

    def test_get_dynamic_dimension_int_keep_const(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        result = g.get_dynamic_dimension(5, keep_const=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqualArray(result, np.array([5], dtype=np.int64))

    def test_get_dynamic_dimension_int_no_keep_const(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        result = g.get_dynamic_dimension(5, keep_const=False)
        self.assertIsInstance(result, str)
        self.assertIn(result, g.initializers_dict)
        self.assertEqualArray(g.initializers_dict[result], np.array([5], dtype=np.int64))

    def test_get_dynamic_dimension_str_rank1(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", TensorProto.FLOAT, (3,), False)
        result = g.get_dynamic_dimension("X", keep_const=True)
        self.assertEqual(result, "X")

    def test_get_dynamic_dimension_str_rank0(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("d", TensorProto.INT64, tuple(), False)
        result = g.get_dynamic_dimension("d", keep_const=True)
        # rank-0 scalar is unsqueezed to rank-1
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "d")

    def test_is_more_precise_both_static_same(self):
        g = GraphBuilder(18)
        self.assertTrue(g.is_more_precise((1, 2), (1, 2)))

    def test_is_more_precise_both_static_different(self):
        g = GraphBuilder(18)
        self.assertFalse(g.is_more_precise((1, 3), (1, 2)))

    def test_is_more_precise_int_over_string(self):
        g = GraphBuilder(18)
        self.assertTrue(g.is_more_precise((1, 2), (1, "d")))

    def test_is_more_precise_string_vs_int(self):
        g = GraphBuilder(18)
        self.assertTrue(g.is_more_precise((1, "d"), (1, 2)))

    def test_is_more_precise_both_dynamic_same(self):
        g = GraphBuilder(18)
        self.assertTrue(g.is_more_precise(("batch", 4), ("batch", 4)))

    def test_is_more_precise_both_dynamic_different(self):
        g = GraphBuilder(18)
        self.assertFalse(g.is_more_precise(("a", 4), ("b", 4)))

    def test_is_more_precise_different_ranks_raises(self):
        g = GraphBuilder(18)
        self.assertRaises(AssertionError, g.is_more_precise, (1, 2), (1, 2, 3))

    def test_add_stat(self):
        g = GraphBuilder(18, ir_version=9)
        # First call creates the kind/name entry with value 1
        g.add_stat("op", "Add")
        self.assertEqual(g.statistics_["op"]["Add"], 1)
        # Second call increments the counter
        g.add_stat("op", "Add")
        self.assertEqual(g.statistics_["op"]["Add"], 2)
        # New name under existing kind
        g.add_stat("op", "Mul")
        self.assertEqual(g.statistics_["op"]["Mul"], 1)
        # New kind
        g.add_stat("pattern", "Reshape")
        self.assertEqual(g.statistics_["pattern"]["Reshape"], 1)
        # Existing kind/name counters are unchanged
        self.assertEqual(g.statistics_["op"]["Add"], 2)

    def test_make_tensor_value_info_from_name(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)

        # Case 1: name has both type and shape
        g.set_type("x", TFLOAT)
        g.set_shape("x", (2, 3))
        vi = g.make_tensor_value_info_from_name("x")
        self.assertEqual(vi.name, "x")
        self.assertEqual(vi.type.tensor_type.elem_type, TFLOAT)
        self.assertEqual([d.dim_value for d in vi.type.tensor_type.shape.dim], [2, 3])

        # Case 2: name has type and rank but no shape
        g.set_type("y", TINT64)
        g.set_rank("y", 3)
        vi = g.make_tensor_value_info_from_name("y")
        self.assertEqual(vi.name, "y")
        self.assertEqual(vi.type.tensor_type.elem_type, TINT64)
        self.assertEqual(len(vi.type.tensor_type.shape.dim), 3)

        # Case 3: name has no type or rank — returns an empty TypeProto value info
        vi = g.make_tensor_value_info_from_name("z")
        self.assertEqual(vi.name, "z")
        self.assertFalse(vi.type.HasField("tensor_type"))

    def test_rename_results_basic(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        nodes = [oh.make_node("Add", ["x", "y"], ["z"], name="n0")]
        replacements = {"x": "x", "y": "y"}
        new_nodes = g._rename_results(nodes, replacements)
        self.assertEqual(len(new_nodes), 1)
        self.assertEqual(list(new_nodes[0].input), ["x", "y"])
        self.assertEqual(list(new_nodes[0].output), ["z"])
        self.assertIn("z", replacements)
        self.assertEqual(replacements["z"], "z")

    def test_rename_results_renamed_inputs(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        nodes = [oh.make_node("Add", ["x", "y"], ["z"], name="n0")]
        replacements = {"x": "new_x", "y": "new_y"}
        new_nodes = g._rename_results(nodes, replacements)
        self.assertEqual(list(new_nodes[0].input), ["new_x", "new_y"])
        self.assertEqual(list(new_nodes[0].output), ["z"])

    def test_rename_results_output_already_in_replacements(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        # Output 'z' is already in replacements with the same value (final output)
        nodes = [oh.make_node("Relu", ["x"], ["z"], name="n0")]
        replacements = {"x": "x", "z": "z"}
        new_nodes = g._rename_results(nodes, replacements)
        self.assertEqual(list(new_nodes[0].output), ["z"])

    def test_rename_results_multiple_nodes(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        # Add(x, y) -> tmp; Relu(tmp) -> out
        nodes = [
            oh.make_node("Add", ["x", "y"], ["tmp"], name="n0"),
            oh.make_node("Relu", ["tmp"], ["out"], name="n1"),
        ]
        replacements = {"x": "new_x", "y": "new_y", "out": "out"}
        new_nodes = g._rename_results(nodes, replacements)
        self.assertEqual(len(new_nodes), 2)
        # Inputs of first node are renamed
        self.assertEqual(list(new_nodes[0].input), ["new_x", "new_y"])
        # Output of first node becomes the input of the second node
        first_out = new_nodes[0].output[0]
        self.assertEqual(list(new_nodes[1].input), [first_out])
        # Final output is preserved
        self.assertEqual(list(new_nodes[1].output), ["out"])

    def test_rename_results_with_graph_attribute(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        # Build a minimal If node with a then_branch subgraph
        then_graph = oh.make_graph(
            [oh.make_node("Add", ["outer_x", "outer_y"], ["branch_out"])],
            "then_branch",
            [],
            [oh.make_tensor_value_info("branch_out", TensorProto.FLOAT, [])],
        )
        if_node = oh.make_node("If", ["cond"], ["result"], name="n0")
        if_node.attribute.append(oh.make_attribute("then_branch", then_graph))
        replacements = {"cond": "cond", "result": "result"}
        new_nodes = g._rename_results([if_node], replacements)
        self.assertEqual(len(new_nodes), 1)
        self.assertEqual(new_nodes[0].op_type, "If")

    def test_rename_results_in_subgraph_no_replacement_needed(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        subgraph = oh.make_graph(
            [oh.make_node("Add", ["a", "b"], ["c"])],
            "sub",
            [],
            [oh.make_tensor_value_info("c", TensorProto.FLOAT, [])],
        )
        # No actual substitution: replacements map each name to itself
        replacements = {"a": "a", "b": "b"}
        result = g._rename_results_in_subgraph(subgraph, replacements=replacements)
        # When nothing changes the original graph object is returned
        self.assertIs(result, subgraph)

    def test_rename_results_in_subgraph_with_replacement(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        subgraph = oh.make_graph(
            [oh.make_node("Add", ["a", "b"], ["c"])],
            "sub",
            [],
            [oh.make_tensor_value_info("c", TensorProto.FLOAT, [])],
        )
        replacements = {"a": "new_a", "b": "b"}
        result = g._rename_results_in_subgraph(subgraph, replacements=replacements)
        self.assertEqual(result.name, "sub")
        self.assertEqual(len(result.node), 1)
        self.assertEqual(list(result.node[0].input), ["new_a", "b"])
        self.assertEqual(list(result.node[0].output), ["c"])

    def test_rename_results_in_subgraph_shadowing(self):
        # Verify that once a node re-defines a name that was being replaced,
        # the replacement stops applying to subsequent nodes.
        g = GraphBuilder(18, ir_version=9, as_function=True)
        subgraph = oh.make_graph(
            [
                oh.make_node("Add", ["a", "b"], ["a"]),  # shadows 'a'
                oh.make_node("Relu", ["a"], ["c"]),
            ],
            "sub",
            [],
            [oh.make_tensor_value_info("c", TensorProto.FLOAT, [])],
        )
        # 'a' should be renamed to 'new_a' only in the first node's inputs
        replacements = {"a": "new_a", "b": "b"}
        result = g._rename_results_in_subgraph(subgraph, replacements=replacements)
        # First node input uses the replacement; output keeps 'a'
        self.assertEqual(list(result.node[0].input), ["new_a", "b"])
        # Second node input must use the local 'a' (shadowed), not 'new_a'
        self.assertEqual(list(result.node[1].input), ["a"])

    def test_empty_copy(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g2 = g.empty_copy()
        self.assertIsInstance(g2, GraphBuilder)
        self.assertIsNot(g, g2)
        self.assertEqual(g.opsets, g2.opsets)

    def test_empty_copy_as_function(self):
        g = GraphBuilder(18, ir_version=9, as_function=False)
        g2 = g.empty_copy(as_function=True)
        self.assertIsInstance(g2, GraphBuilder)
        self.assertTrue(g2.as_function)

    def test_empty_copy_shapable_false(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g2 = g.empty_copy(_shapable=False)
        self.assertIsInstance(g2, GraphBuilder)
        self.assertFalse(g2._debug_shape_missing)

    def test_pretty_tensor_with_shape(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = g.pretty_tensor(arr)
        self.assertIn("float32", result)
        self.assertIn("2", result)

    def test_pretty_tensor_without_shape(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)

        class NoShape:
            pass

        result = g.pretty_tensor(NoShape())
        self.assertIn("no pretty", result)

    def test_pretty_node_shape_op(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        node = oh.make_node("Shape", ["X"], ["shape_out"])
        result = g.pretty_node(node)
        self.assertIn("Shape", result)
        self.assertIn("X", result)
        self.assertIn("shape_out", result)

    def test_pretty_node_shape_op_with_attributes(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        node = oh.make_node("Shape", ["X"], ["shape_out"], start=1, end=3)
        result = g.pretty_node(node)
        self.assertIn("Shape", result)
        self.assertIn("start=1", result)
        self.assertIn("end=3", result)

    def test_pretty_node_shape_true(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        node = oh.make_node("Add", ["X", "Y"], ["Z"])
        g.set_type("X", TFLOAT)
        g.set_shape("X", (2, 3))
        g.set_type("Y", TFLOAT)
        g.set_shape("Y", (2, 3))
        g.set_type("Z", TFLOAT)
        g.set_shape("Z", (2, 3))
        result = g.pretty_node(node, shape=True)
        self.assertIn("X:1|2x3", result)
        self.assertIn("Y:1|2x3", result)
        self.assertIn("Z:1|2x3", result)
        self.assertIn("->", result)

    def test_pretty_node_shape_op_with_shape_true(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        node = oh.make_node("Shape", ["X"], ["shape_out"])
        g.set_type("X", TFLOAT)
        g.set_shape("X", (3, 4))
        g.set_type("shape_out", TINT64)
        g.set_shape("shape_out", (2,))
        result = g.pretty_node(node, shape=True)
        self.assertIn("Shape", result)
        self.assertIn("X:1|3x4", result)
        self.assertIn("shape_out:7|2", result)
        self.assertIn("->", result)

    def test_do_not_turn_constant_initializers_flag_set(self):
        # When _do_not_turn_constant_initializers is True (set by move_initializers_to_constant),
        # the method must return True regardless of name.
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", TFLOAT, (2, 4), False)
        np_weights = np.ones((4, 3), dtype=np.float32)
        g.make_initializer("weights", np_weights)
        g.move_initializers_to_constant(full_parameter_name=False)
        self.assertTrue(g.do_not_turn_constant_initializers_maybe_because_of_showing("weights"))
        self.assertTrue(g.do_not_turn_constant_initializers_maybe_because_of_showing("unknown"))

    def test_do_not_turn_constant_initializers_no_parent(self):
        # Without a parent the method always returns False.
        g = GraphBuilder(18, ir_version=9)
        g.make_initializer("cst", np.ones((2, 2), dtype=np.float32))
        self.assertFalse(g.do_not_turn_constant_initializers_maybe_because_of_showing("cst"))
        self.assertFalse(g.do_not_turn_constant_initializers_maybe_because_of_showing("other"))

    def test_do_not_turn_constant_initializers_parent_does_not_have_name(self):
        # Parent does not know the name at all: returns False.
        parent = GraphBuilder(18, ir_version=9)
        child = GraphBuilder(18, ir_version=9, _parent=parent)
        child.make_initializer("cst", np.arange(4).reshape((2, 2)).astype(np.float32))
        self.assertFalse(child.do_not_turn_constant_initializers_maybe_because_of_showing("cst"))

    def test_do_not_turn_constant_initializers_same_constant_in_parent(self):
        # Same constant in both parent and child: has_exact_same_constant_in_context returns True,
        # so the method returns not True = False (safe to share with parent).
        parent = GraphBuilder(18, ir_version=9)
        np_cst = np.arange(4).reshape((2, 2)).astype(np.float32)
        parent.make_initializer("cst", np_cst)

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        child.make_initializer("cst", np_cst)

        self.assertFalse(child.do_not_turn_constant_initializers_maybe_because_of_showing("cst"))

    def test_do_not_turn_constant_initializers_different_constant_in_parent(self):
        # Different values for same name: has_exact_same_constant_in_context returns False,
        # so the method returns not False = True (shadowing would occur).
        parent = GraphBuilder(18, ir_version=9)
        np_cst1 = np.arange(4).reshape((2, 2)).astype(np.float32)
        parent.make_initializer("cst", np_cst1)

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        np_cst2 = np_cst1 + 1.0
        child.make_initializer("cst", np_cst2)

        self.assertTrue(child.do_not_turn_constant_initializers_maybe_because_of_showing("cst"))

    def test_do_not_turn_constant_initializers_large_constant_recurse_to_parent(self):
        # Large constants (>= 128 elements) cause has_exact_same_constant_in_context to return
        # None, so the method recurses to the parent. The parent has no parent, so it returns
        # False.
        parent = GraphBuilder(18, ir_version=9)
        np_cst = np.arange(128).reshape((16, 8)).astype(np.float32)
        parent.make_initializer("cst", np_cst)

        child = GraphBuilder(18, ir_version=9, _parent=parent)
        child.make_initializer("cst", np_cst)

        self.assertFalse(child.do_not_turn_constant_initializers_maybe_because_of_showing("cst"))


class TestPositionMsg(ExtTestCase):
    def _make_simple_model(self):
        return oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Abs", ["X"], ["a"], name="n0"),
                    oh.make_node("Neg", ["a"], ["b"], name="n1"),
                    oh.make_node("Relu", ["b"], ["Y"], name="n2"),
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, [None])],
                [oh.make_tensor_value_info("Y", TFLOAT, [None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )

    def test_position_msg_no_around(self):
        model = self._make_simple_model()
        gr = GraphBuilder(model)
        msg = gr._position_msg(gr.nodes)
        self.assertIsInstance(msg, str)
        self.assertIn("Abs", msg)
        self.assertIn("Neg", msg)
        self.assertIn("Relu", msg)
        # input/output position lines should appear
        self.assertIn("pos(", msg)

    def test_position_msg_single_node(self):
        model = self._make_simple_model()
        gr = GraphBuilder(model)
        node = gr.nodes[1]  # Neg node
        msg = gr._position_msg([node])
        self.assertIsInstance(msg, str)
        self.assertIn("Neg", msg)
        self.assertIn("pos(a)", msg)
        self.assertIn("pos(b)", msg)

    def test_position_msg_with_around(self):
        model = self._make_simple_model()
        gr = GraphBuilder(model)
        node = gr.nodes[1]  # Neg node
        msg = gr._position_msg([node], around=(1, 1))
        self.assertIsInstance(msg, str)
        self.assertIn("Neg", msg)
        # context window separator should be present
        self.assertIn("---", msg)
        # positional prefix for context nodes
        self.assertIn("P", msg)

    def test_position_msg_with_none_node(self):
        model = self._make_simple_model()
        gr = GraphBuilder(model)
        # None entries in the list should be skipped without error
        msg = gr._position_msg([None, gr.nodes[0]])
        self.assertIsInstance(msg, str)
        self.assertIn("Abs", msg)

    def test_value_info_static_shapes(self):
        """Shape info for intermediate tensors must be added even when there are no
        dynamic dimensions (regression test for early-return bug in
        _add_shape_information)."""
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, (2, 3))
        g.make_node("Relu", ["X"], ["tmp"], name="relu1")
        g.make_node("Relu", ["tmp"], ["output_0"], name="relu2")
        g.make_tensor_output("output_0", TFLOAT, (2, 3), indexed=True)
        onx = g.to_onnx(optimize=False)
        vi_names = {vi.name for vi in onx.graph.value_info}
        self.assertIn("tmp", vi_names, "intermediate tensor 'tmp' must have shape info")
        tmp_vi = next(vi for vi in onx.graph.value_info if vi.name == "tmp")
        shape = [d.dim_value for d in tmp_vi.type.tensor_type.shape.dim]
        self.assertEqual(shape, [2, 3])

    def test_value_info_dynamic_shapes(self):
        """Shape info for intermediate tensors must also be present with dynamic dims."""
        g = GraphBuilder(18, ir_version=9, dynamic_shapes={"X": {0: "batch"}})
        g.make_tensor_input("X", TFLOAT, ("batch", 3))
        g.make_node("Relu", ["X"], ["tmp"], name="relu1")
        g.make_node("Relu", ["tmp"], ["output_0"], name="relu2")
        g.make_tensor_output("output_0", TFLOAT, ("batch", 3), indexed=True)
        onx = g.to_onnx(optimize=False)
        vi_names = {vi.name for vi in onx.graph.value_info}
        self.assertIn("tmp", vi_names, "intermediate tensor 'tmp' must have shape info")
        tmp_vi = next(vi for vi in onx.graph.value_info if vi.name == "tmp")
        shape = [d.dim_param or d.dim_value for d in tmp_vi.type.tensor_type.shape.dim]
        self.assertEqual(shape, ["batch", 3])

    def test_optimize_applies_dynamic_dimension_renaming(self):
        """optimize() calls _improves_dynamic_dimension_naming(apply_replacements=True),
        which must update get_shape() and get_shape_renamed() to reflect user-visible
        dimension names instead of internal symbolic tokens."""
        # Build a minimal single-Relu graph with internal dim tokens "s0" / "s1".
        g = GraphBuilder(
            18, ir_version=9, optimization_options=OptimizationOptions(passes=[], patterns=None)
        )
        g.make_tensor_input("X", TFLOAT, ("s0", "s1"))
        g.make_node("Relu", ["X"], ["Y"], name="A")
        g.make_tensor_output("Y", TFLOAT, ("s0", "s1"), indexed=False)

        # Register bidirectional constraints linking internal tokens to user names.
        g.add_to_constraints("s0", "batch")
        g.add_to_constraints("batch", "s0")
        g.add_to_constraints("s1", "seq")
        g.add_to_constraints("seq", "s1")

        # Declare the user-visible names so that _improves_dynamic_dimension_naming
        # treats "batch" and "seq" as the preferred (original) names.
        g.dynamic_dimensions_source["batch"] = [{"input_name": "X", "axis": 0}]
        g.dynamic_dimensions_source["seq"] = [{"input_name": "X", "axis": 1}]

        # Verify shapes use internal tokens before optimization.
        self.assertEqual(g.get_shape("X"), ("s0", "s1"))
        self.assertEqual(g.get_shape("Y"), ("s0", "s1"))

        # Run optimize()
        # this triggers _improves_dynamic_dimension_naming(apply_replacements=True).
        g.optimize()

        # After optimize(), get_shape() must reflect the user-visible names.
        self.assertEqual(g.get_shape("X"), ("batch", "seq"))
        self.assertEqual(g.get_shape("Y"), ("batch", "seq"))

        # get_shape_renamed() must also return the user-visible names.
        self.assertEqual(g.get_shape_renamed("X"), ("batch", "seq"))
        self.assertEqual(g.get_shape_renamed("Y"), ("batch", "seq"))

    def test_no_duplicate_batch_names_multiple_outputs(self):
        """DYN dimensions at axis 0 that are all constrained equal to the
        user-defined 'batch' name must not generate 'batch_1', 'batch_2', etc.
        This covers the arnir0/Tiny-LLM case where multiple outputs share the
        same batch dimension but are tracked under separate DYN names."""
        g = GraphBuilder(18, ir_version=9)

        # One input with a user-visible 'batch' dimension.
        g.make_tensor_input("X", TFLOAT, ("batch", 4))

        # Three outputs sharing the same batch dimension, recorded under
        # separate internal DYN names.
        g.make_node("Relu", ["X"], ["Y0"], name="relu0")
        g.make_node("Relu", ["X"], ["Y1"], name="relu1")
        g.make_node("Relu", ["X"], ["Y2"], name="relu2")
        g.set_shape("Y0", ("DYN0", 4))
        g.set_shape("Y1", ("DYN1", 4))
        g.set_shape("Y2", ("DYN2", 4))
        g.set_type("Y0", TFLOAT)
        g.set_type("Y1", TFLOAT)
        g.set_type("Y2", TFLOAT)
        g.make_tensor_output("Y0", TFLOAT, ("DYN0", 4), indexed=False)
        g.make_tensor_output("Y1", TFLOAT, ("DYN1", 4), indexed=False)
        g.make_tensor_output("Y2", TFLOAT, ("DYN2", 4), indexed=False)

        # Declare all names in dynamic_dimensions_source.
        g.dynamic_dimensions_source["batch"] = [{"input_name": "X", "axis": 0}]
        g.dynamic_dimensions_source["DYN0"] = [{"input_name": "Y0", "axis": 0}]
        g.dynamic_dimensions_source["DYN1"] = [{"input_name": "Y1", "axis": 0}]
        g.dynamic_dimensions_source["DYN2"] = [{"input_name": "Y2", "axis": 0}]

        # Register bidirectional constraints: DYN0, DYN1, DYN2 all equal to batch.
        for dyn in ("DYN0", "DYN1", "DYN2"):
            g.add_to_constraints("batch", dyn)
            g.add_to_constraints(dyn, "batch")

        replacements = g._improves_dynamic_dimension_naming()

        # Each DYN dimension must be renamed to "batch", never to "batch_1" etc.
        for dyn in ("DYN0", "DYN1", "DYN2"):
            got = replacements.get(dyn)
            self.assertEqual(
                got,
                "batch",
                f"Expected {dyn!r} → 'batch' but got {got!r}. "
                "Redundant 'batch_N' aliases were created.",
            )

        # Output shapes must show "batch" at axis 0.
        for out_name in ("Y0", "Y1", "Y2"):
            renamed = g.get_shape_renamed(out_name)
            self.assertEqual(
                renamed[0],
                "batch",
                f"Output {out_name!r} axis-0 dim should be 'batch', got {renamed[0]!r}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
