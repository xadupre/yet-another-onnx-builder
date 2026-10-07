import unittest
from typing import Dict, List
import onnx_light
import onnx_light.onnx.helper as oh
import numpy as np
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import AttributeProto, FunctionProto, TensorProto, ValueInfoProto
from onnx_light.onnx_core.shape_inference import SymShape
from packaging.version import Version
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

        gr.inline_functions(verbose=1)
        function_proto = gr.to_onnx(
            function_options=FunctionOptions(export_as_function=True, name="lr"), inline=False
        )
        self.assertIsInstance(function_proto, ExportArtifact)
        self.assertIsInstance(function_proto.proto, FunctionProto)
        self.assertEqual(function_proto.proto.domain, "")
        self.assertEqual(function_proto.proto.name, "lr")
        got = ExtendedReferenceEvaluator(function_proto.proto).run(None, feeds)[0]
        self.assertEqualArray(expected, got)
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
        g.make_tensor_input("X", TFLOAT, (None, 4))
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
        self.assertEqualArray(expected, got[0], atol=1e-6, rtol=1e-6)

    @ignore_warnings(DeprecationWarning)
    def test_as_function_constant_full(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", TFLOAT, (None, 4))
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
        self.assertEqualArray(expected, got[0], atol=1e-6, rtol=1e-6)

    @ignore_warnings(DeprecationWarning)
    def test_as_function_second(self):
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", TFLOAT, (None, 4))
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
        np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 1000

        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", indexed=False)
        self.assertEqualArray(gf.get_constant("weights"), np_weights)

        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", TFLOAT, (None, 4))
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
        self.assertEqualArray(g.get_constant("weights"), np_weights)

        bias2 = g.make_initializer("bias2", np_bias2)
        g.op.Add(
            g.anyop.Regression("X", *new_inits, name="linear", domain="custom"),
            bias2,
            outputs=["Y"],
        )
        g.make_tensor_output("Y", indexed=False)
        nodes = [(node.domain, node.op_type, node.input, node.output) for node in g.nodes]
        regression_output = str(nodes[0][3][0])
        self.assertEqual(
            nodes,
            [
                ("custom", "Regression", ["X", "weights", "bias"], [regression_output]),
                ("", "Add", [regression_output, "bias2"], ["Y"]),
            ],
        )

        # finally, the conversion to onnx
        text = g.pretty_text()
        self.assertIn(regression_output, text)
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
        self.assertEqual(set(fct.function.initializers_name), {"weights", "bias", "bias2"})
        self.assertIsInstance(fct.function.initializers_dict, dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct.function.initializers_dict.values())
        )
        self.assertEqual(len(fct.function.initializers_name), len(fct.function.initializers_dict))
        proto = fct.proto
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(proto.input, ["X", *fct.function.initializers_name])
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
        self.assertEqualArray(expected, got[0], atol=2e-5)

    @ignore_warnings(DeprecationWarning)
    def test_as_function_nested_unique(self):
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
        np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 100
        np_bias3 = np.arange(3).reshape((1, 3)).astype(np.float32) + 1000

        # first function
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", TFLOAT, (None, 4))
        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", indexed=False)
        self.assertEqualArray(gf.get_constant("weights"), np_weights)

        # second function calling the first one
        g2 = GraphBuilder(18, ir_version=9, as_function=True)
        g2.make_tensor_input("X", TFLOAT, (None, 4))
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
        g.make_tensor_input("X", TFLOAT, (None, 4))
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
        self.assertIn("RegressionBias", g.pretty_text())

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
        self.assertEqual(
            set(fct.function.initializers_name), {"weights", "bias", "bias2", "bias3"}
        )
        self.assertIsInstance(fct.function.initializers_dict, dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct.function.initializers_dict.values())
        )
        self.assertEqual(len(fct.function.initializers_name), len(fct.function.initializers_dict))
        proto = fct.proto
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(proto.input, ["X", *fct.function.initializers_name])
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
        self.assertEqual(f2.input, ["X", *new_inits])

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        feeds.update(fct.function.initializers_dict)
        self.assertEqualArray(np_weights, feeds["weights"])
        self.assertEqualArray(np_bias, feeds["bias"])
        self.assertEqualArray(np_bias2, feeds["bias2"])
        self.assertEqualArray(np_bias3, feeds["bias3"])
        self.assertEqual(set(feeds), {"X", "weights", "bias", "bias3", "bias2"})
        expected = feeds["X"] @ np_weights + np_bias + np_bias2 + np_bias3

        # Evaluation of a function
        self.assertEqual(g.opsets[""], 18)
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
        gf.make_tensor_input("X", TFLOAT, (None, 4))
        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", indexed=False)
        self.assertEqualArray(gf.get_constant("weights"), np_weights)

        # main graph
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", TFLOAT, (None, 4))
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
        self.assertEqualArray(g.get_constant("weights"), np_weights)

        # function 3: the same name but different
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", TFLOAT, (None, 4))

        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Sub(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", indexed=False)
        self.assertEqualArray(gf.get_constant("weights"), np_weights)

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
        self.assertEqualArray(g.get_constant("weights"), np_weights)

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
        self.assertEqual((f2.domain, f2.name), (domain_name, function_name))
        self.assertNotEqual(f1.name, f2.name)
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
            gf.make_tensor_input("X", TFLOAT, (None, 4))
            init = gf.make_initializer("weights", np_weights)
            bias = gf.make_initializer("bias", np_bias)
            gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
            gf.make_tensor_output("Y", indexed=False)
            self.assertEqualArray(gf.get_constant("weights"), np_weights)

            # second function calling the first one
            g2 = GraphBuilder(18, ir_version=9, as_function=True)
            g2.make_tensor_input("X", TFLOAT, (None, 4))
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
        g.make_tensor_input("X", TFLOAT, (None, 4))

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
        self.assertEqual(len(g.functions), 3)

        g.op.Add(
            g.anyop.RegressionBias("X", *new_inits_1, name="reg2", domain="custom"),
            g.make_node(function_name, ["X", *new_inits_2], name="reg2", domain=domain_name),
            outputs=["Y"],
        )
        g.make_tensor_output("Y", indexed=False)

        # finally, the conversion to onnx
        self.assertIn("RegressionBias", g.pretty_text())

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
        self.assertEqual(set(fct.function.initializers_name), {"weights", "bias", "bias2"})
        self.assertIsInstance(fct.function.initializers_dict, dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct.function.initializers_dict.values())
        )
        self.assertEqual(len(fct.function.initializers_name), len(fct.function.initializers_dict))
        proto = fct.proto
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(proto.input, ["X", *fct.function.initializers_name])
        self.assertEqual(proto.domain, "mine")
        self.assertEqual(proto.name, "linear")
        self.assertEqual(3, len(fct.function.nested_functions))
        f1 = fct.function.nested_functions[0]
        self.assertEqual(f1.domain, "custom")
        self.assertEqual(f1.name, "Regression")
        self.assertEqual(f1.output, ["Y"])
        self.assertEqual(f1.input, ["X", "weights", "bias"])
        f2 = fct.function.nested_functions[1]
        self.assertEqual(f2.domain, "custom")
        self.assertEqual(f2.name, "RegressionBias")
        self.assertEqual(f2.output, ["Y"])
        self.assertEqual(f2.input, ["X", *new_inits_1])
        f3 = fct.function.nested_functions[2]
        self.assertEqual((f3.domain, f3.name), (domain_name, function_name))
        self.assertNotEqual(f2.name, f3.name)
        self.assertEqual(f3.input, ["X", *new_inits_2])
        for function in (f2, f3):
            calls = [node for node in function.node if node.domain == f1.domain]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].op_type, f1.name)

        feeds = dict(X=np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32))
        feeds.update(fct.function.initializers_dict)
        self.assertEqualArray(np_weights, feeds["weights"])
        self.assertEqualArray(np_bias, feeds["bias"])
        self.assertEqualArray(np_bias2, feeds["bias2"])
        expected = (feeds["X"] @ np_weights + np_bias + np_bias2) * 2

        # Evaluation of a function
        self.assertEqual(g.opsets[""], 18)
        ref = ExtendedReferenceEvaluator(fct.proto, functions=fct.function.nested_functions)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], rtol=1e-6)

        # Same with a model
        proto = g.to_onnx(inline=False)
        self.assertEqual(len(proto.functions), 3)
        ref = ExtendedReferenceEvaluator(proto)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], rtol=1e-6)

    @ignore_warnings(DeprecationWarning)
    def test_as_function_nested_twice_merge(self):

        def _make_function():
            np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
            np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
            np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 100

            # first function
            gf = GraphBuilder(18, ir_version=9, as_function=True)
            gf.make_tensor_input("X", TFLOAT, (None, 4))
            init = gf.make_initializer("weights", np_weights)
            bias = gf.make_initializer("bias", np_bias)
            gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
            gf.make_tensor_output("Y", indexed=False)
            self.assertEqualArray(gf.get_constant("weights"), np_weights)

            # second function calling the first one
            g2 = GraphBuilder(18, ir_version=9, as_function=True)
            g2.make_tensor_input("X", TFLOAT, (None, 4))
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
        g.make_tensor_input("X", TFLOAT, (None, 4))

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
        self.assertIn("RegressionBias", g.pretty_text())

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
        self.assertEqual(set(fct.function.initializers_name), {"weights", "bias", "bias2"})
        self.assertIsInstance(fct.function.initializers_dict, dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct.function.initializers_dict.values())
        )
        self.assertEqual(len(fct.function.initializers_name), len(fct.function.initializers_dict))
        proto = fct.proto
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(proto.input, ["X", *fct.function.initializers_name])
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
        self.assertEqual(f2.input, ["X", *new_inits_1])
        self.assertEqual((domain_name, function_name), (f2.domain, f2.name))
        self.assertEqual(new_inits_1, new_inits_2)

        feeds = dict(X=np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32))
        feeds.update(fct.function.initializers_dict)
        self.assertEqualArray(np_weights, feeds["weights"])
        self.assertEqualArray(np_bias, feeds["bias"])
        self.assertEqualArray(np_bias2, feeds["bias2"])
        expected = (feeds["X"] @ np_weights + np_bias + np_bias2) * 2

        # Evaluation of a function
        self.assertEqual(g.opsets[""], 18)
        ref = ExtendedReferenceEvaluator(fct.proto, functions=fct.function.nested_functions)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], rtol=1e-6)

        # Same with a model
        proto = g.to_onnx(inline=False)
        self.assertEqual(len(proto.functions), 2)
        ref = ExtendedReferenceEvaluator(proto)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], rtol=1e-6)

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
        self.assertEqual(g.get_shape("d"), (None, None, None))
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
                g = GraphBuilder(18)
                g.make_tensor_input("X", TFLOAT, s1)
                g.op.Reshape("X", np.array(s2, dtype=np.int64), outputs=["Y"])
                self.assertEqual(SymShape(expected), g.shapes_context.get("Y").shape)

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
        gr.inner_builder.move_shape_and_size_nodes()
        onx = gr.to_onnx()
        available = {str(value.name) for value in onx.graph.input} | {
            str(value.name) for value in onx.graph.initializer
        }
        for node in onx.graph.node:
            self.assertTrue(set(map(str, node.input)) - {""} <= available)
            available.update(map(str, node.output))
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

        gr.inline_functions(verbose=1)
        function_proto = gr.to_onnx(
            function_options=FunctionOptions(export_as_function=True, name="lr"), inline=False
        )
        self.assertIsInstance(function_proto, ExportArtifact)
        self.assertIsInstance(function_proto.proto, FunctionProto)
        self.assertEqual(function_proto.proto.domain, "")
        self.assertEqual(function_proto.proto.name, "lr")
        got = ExtendedReferenceEvaluator(function_proto.proto).run(None, feeds)[0]
        self.assertEqualArray(expected, got)
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
                value = ValueInfoProto()
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
    @unittest.skipIf(
        Version(onnx_light.__version__) <= Version("0.1.30"), "xadupre/onnx-light#5192"
    )
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
                        oh.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3]),
                        oh.make_tensor_value_info("A", TensorProto.FLOAT, [3, 3]),
                        oh.make_tensor_value_info("B", TensorProto.FLOAT, [3, 3]),
                    ],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3])],
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
        self.assertTrue(all(node is not None for node in gr.nodes))
        self.assertEqual(len(gr.functions), 2)
        onx = gr.to_onnx(inline=False)
        self.assertTrue(all(node is not None for node in gr.nodes))
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
                value = ValueInfoProto()
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
    @unittest.skipIf(
        Version(onnx_light.__version__) <= Version("0.1.30"), "xadupre/onnx-light#5192"
    )
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
                        oh.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3]),
                        oh.make_tensor_value_info("A", TensorProto.FLOAT, [3, 3]),
                        oh.make_tensor_value_info("B", TensorProto.FLOAT, [3, 3]),
                    ],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3])],
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
        self.assertTrue(all(node is not None for node in gr.nodes))
        self.assertEqual(len(gr.functions), 2)
        onx = gr.to_onnx(inline=False)
        self.assertTrue(all(node is not None for node in gr.nodes))
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

    def test_update_model_with_parameter_renaming(self):
        """Test _update_model_with_parameter_renaming renames initializer in nodes."""
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, (2, 4))
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32)
        w_init = g.make_initializer("p_layer_weight", np_weights, parameter_name="layer.weight")
        self.assertEqual(w_init, "layer.weight")
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
        self.assertEqualArray(feeds["X"] @ np_weights, got, atol=1e-5, rtol=1e-6)

    def test_update_model_with_parameter_renaming_multiple(self):
        """Test _update_model_with_parameter_renaming with multiple renamed parameters."""
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, (2, 4))
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32)
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10.0
        w_init = g.make_initializer("p_w", np_weights, parameter_name="fc.weight")
        b_init = g.make_initializer("p_b", np_bias, parameter_name="fc.bias")
        self.assertEqual((w_init, b_init), ("fc.weight", "fc.bias"))
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
        self.assertEqualArray(feeds["X"] @ np_weights + np_bias, got, atol=1e-5, rtol=1e-6)


@requires_torch()
class TestGetInputDynamicShape(ExtTestCase):
    def setUp(self):
        self.g = GraphBuilder(18, ir_version=9)

    def test_is_sequence_false_for_non_sequence(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, (3, 4))
        self.assertFalse(g.is_sequence("X"))

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

    def _make_node_with_attrs(self, **attrs):
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        for name, value in attrs.items():
            node.attribute.append(oh.make_attribute(name, value))
        return node


class TestGraphBuilderGetTypeKnown(ExtTestCase):
    def native_alias_cleanup(self, nodes, replacements):
        """Exercises native reference rewriting by removing explicit input aliases."""
        produced = {str(name) for node in nodes for name in node.output}
        aliases = [
            oh.make_node("Identity", [new], [old])
            for old, new in replacements.items()
            if old != new and old not in produced
        ]
        inputs = [
            oh.make_tensor_value_info(name, TensorProto.BOOL if name == "cond" else TFLOAT, [])
            for name in sorted(set(replacements.values()) - produced)
        ]
        outputs = [oh.make_tensor_value_info(str(name), TFLOAT, []) for name in nodes[-1].output]
        model = oh.make_model(
            oh.make_graph(aliases + nodes, "aliases", inputs, outputs),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        builder = GraphBuilder(model)
        builder.inner_builder.remove_identity_nodes()
        return builder.nodes

    def native_subgraph_alias_cleanup(self, graph, replacements):
        """Exercises native alias rewriting across If branch captures."""
        node = oh.make_node("If", ["cond"], ["result"], then_branch=graph, else_branch=graph)
        nodes = self.native_alias_cleanup([node], {"cond": "cond", **replacements})
        return next(
            attribute.g for attribute in nodes[0].attribute if attribute.name == "then_branch"
        )

    def test_has_exact_same_constant_in_context_same(self):
        # Child and parent have identical small constants: should return True.
        parent = GraphBuilder(18, ir_version=9)
        np_cst = np.arange(4).reshape((2, 2)).astype(np.float32)
        parent.make_initializer("cst", np_cst)

        child = parent.empty_copy()
        child.make_initializer("cst", np_cst)

        result = child.constant_is_equal_to("cst", parent.get_constant("cst"))
        self.assertTrue(result)

    def test_has_exact_same_constant_in_context_different_values(self):
        # Same shape/type but different values: should return False.
        parent = GraphBuilder(18, ir_version=9)
        np_cst1 = np.arange(4).reshape((2, 2)).astype(np.float32)
        parent.make_initializer("cst", np_cst1)

        child = parent.empty_copy()
        np_cst2 = np_cst1 + 1.0
        child.make_initializer("cst", np_cst2)

        result = child.constant_is_equal_to("cst", parent.get_constant("cst"))
        self.assertFalse(result)

    def test_has_exact_same_constant_in_context_different_shape(self):
        # Same name but different shapes: should return False.
        parent = GraphBuilder(18, ir_version=9)
        parent.make_initializer("cst", np.arange(6).reshape((2, 3)).astype(np.float32))

        child = parent.empty_copy()
        child.make_initializer("cst", np.arange(4).reshape((2, 2)).astype(np.float32))

        result = child.constant_is_equal_to("cst", parent.get_constant("cst"))
        self.assertFalse(result)

    def test_has_exact_same_constant_in_context_different_type(self):
        # Same name and shape but different dtypes: should return False.
        parent = GraphBuilder(18, ir_version=9)
        np_cst = np.arange(4).reshape((2, 2)).astype(np.float32)
        parent.make_initializer("cst", np_cst)

        child = parent.empty_copy()
        child.make_initializer("cst", np_cst.astype(np.float64))

        result = child.constant_is_equal_to("cst", parent.get_constant("cst"))
        self.assertFalse(result)

    def test_has_exact_same_constant_in_context_large(self):
        # Native constant values remain comparable above the old cache size limit.
        parent = GraphBuilder(18, ir_version=9)
        np_cst = np.arange(128).reshape((16, 8)).astype(np.float32)
        parent.make_initializer("cst", np_cst)

        child = parent.empty_copy()
        child.make_initializer("cst", np_cst)

        self.assertTrue(child.constant_is_equal_to("cst", parent.get_constant("cst")))
        self.assertFalse(child.constant_is_equal_to("cst", parent.get_constant("cst") + 1))

    def test_has_exact_same_constant_in_context_not_in_child(self):
        # Name is only a constant in the parent, not in the child: should return False.
        parent = GraphBuilder(18, ir_version=9)
        parent.make_initializer("cst", np.arange(4).reshape((2, 2)).astype(np.float32))

        child = parent.empty_copy()

        result = child.constant_is_equal_to("cst", parent.get_constant("cst"))
        self.assertFalse(result)

    def test_has_exact_same_constant_in_context_not_in_parent(self):
        # Name is a constant in the child but not in the parent: should return False.
        parent = GraphBuilder(18, ir_version=9)

        child = parent.empty_copy()
        child.make_initializer("cst", np.arange(4).reshape((2, 2)).astype(np.float32))

        result = parent.constant_is_equal_to("cst", child.get_constant("cst"))
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
        gf.make_tensor_input("X", TFLOAT, (2, 4), False)
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
        g.set_shape("X", (3, 4))
        g.set_shape("Y", (3, 4))
        self.assertEqual(g.shapes_context.get("X").shape, g.shapes_context.get("Y").shape)

    def test_same_shape_static_different(self):
        g = GraphBuilder(18)
        g.set_shape("X", (3, 4))
        g.set_shape("Y", (3, 5))
        self.assertNotEqual(g.shapes_context.get("X").shape, g.shapes_context.get("Y").shape)

    def test_same_shape_different_rank(self):
        g = GraphBuilder(18)
        g.set_shape("X", (3, 4))
        g.set_shape("Y", (3, 4, 5))
        self.assertNotEqual(g.shapes_context.get("X").shape, g.shapes_context.get("Y").shape)

    def test_same_shape_dynamic_same_dim(self):
        g = GraphBuilder(18)
        g.set_shape("X", ("batch", 4))
        g.set_shape("Y", ("batch", 4))
        self.assertEqual(g.shapes_context.get("X").shape, g.shapes_context.get("Y").shape)

    def test_same_shape_dynamic_linked_by_constraints(self):
        g = GraphBuilder(18)
        g.set_shape("X", ("a", 4))
        g.set_shape("Y", ("b", 4))
        self.assertTrue(g.shapes_context.add_constraint("a", "b"))
        self.assertFalse(g.shapes_context.add_constraint("b", "a"))
        self.assertTrue(g.shapes_context.has_constraint("a", "b"))

    def test_same_shape_dynamic_no_constraints(self):
        g = GraphBuilder(18)
        g.set_shape("X", ("a", 4))
        g.set_shape("Y", ("b", 4))
        self.assertNotEqual(g.shapes_context.get("X").shape, g.shapes_context.get("Y").shape)
        self.assertFalse(g.shapes_context.has_constraint("a", "b"))

    def test_set_value_shape_constraint_dim_registration(self):
        # When a name already has a symbolic (string) value shape like ("batch",)
        # and set_value_shape is called with a concrete (int,) tuple,
        # the constraint should be registered for the symbolic dim name ("batch"),
        # not for the literal string "existing".
        g = GraphBuilder(18)
        g.make_tensor_input("X", TFLOAT, ("batch",))
        g.op.Shape("X", outputs=["batch_value"])
        self.assertEqual(g.value_as_shape("batch_value"), ("batch",))
        g.set_value_shape("batch_value", (5,))
        self.assertEqual(g.value_as_shape("batch_value"), (5,))
        self.assertEqual(g.get_shape("X"), ("batch",))

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
        # The native input descriptor supplies the dimension source.
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
        self.assertRaises(ValueError, gr.get_dimension_as_result, "missing_dimension")

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

        # Large arrays must compare their values too.
        large = np.arange(30, dtype=np.float32)
        g.make_initializer("large", large)
        large_same = large.copy()
        self.assertTrue(g.constant_is_equal_to("large", large_same))
        large_different = np.zeros(30, dtype=np.float32)
        self.assertFalse(g.constant_is_equal_to("large", large_different))

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
        self.assertEqualArray(g.get_constant(result), np.array([5], dtype=np.int64))

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

    def test_native_shape_both_static_same(self):
        self.assertEqual(SymShape([1, 2]), SymShape([1, 2]))

    def test_native_shape_both_static_different(self):
        self.assertNotEqual(SymShape([1, 3]), SymShape([1, 2]))

    def test_native_shape_static_information(self):
        self.assertTrue(SymShape([1, 2]).is_fully_known())
        self.assertFalse(SymShape([1, "d"]).is_fully_known())

    def test_native_shape_symbol_is_not_static(self):
        self.assertNotEqual(SymShape([1, "d"]), SymShape([1, 2]))

    def test_native_shape_both_dynamic_same(self):
        self.assertEqual(SymShape(["batch", 4]), SymShape(["batch", 4]))

    def test_native_shape_both_dynamic_different(self):
        self.assertNotEqual(SymShape(["a", 4]), SymShape(["b", 4]))

    def test_native_shape_different_ranks(self):
        self.assertNotEqual(SymShape([1, 2]), SymShape([1, 2, 3]))
        self.assertEqual(SymShape([1, 2, 3]).rank(), 3)

    def test_add_stat(self):
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, (2, 3))
        g.op.Transpose(g.op.Transpose("X", perm=[1, 0]), perm=[1, 0], outputs=["Y"])
        g.make_tensor_output("Y")
        report = g.to_onnx(return_optimize_report=True).report
        self.assertEqual(report.extra["backend"], "onnx-light")
        self.assertEqual(report.extra["rewrites"], 1)
        self.assertEqual(report.stats[0]["pattern"], "TransposeTranspose")

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
        nodes = [oh.make_node("Add", ["x", "y"], ["z"], name="n0")]
        replacements = {"x": "x", "y": "y"}
        new_nodes = self.native_alias_cleanup(nodes, replacements)
        self.assertEqual(len(new_nodes), 1)
        self.assertEqual(list(new_nodes[0].input), ["x", "y"])
        self.assertEqual(list(new_nodes[0].output), ["z"])
        self.assertEqual(replacements, {"x": "x", "y": "y"})

    def test_rename_results_renamed_inputs(self):
        nodes = [oh.make_node("Add", ["x", "y"], ["z"], name="n0")]
        replacements = {"x": "new_x", "y": "new_y"}
        new_nodes = self.native_alias_cleanup(nodes, replacements)
        self.assertEqual(list(new_nodes[0].input), ["new_x", "new_y"])
        self.assertEqual(list(new_nodes[0].output), ["z"])

    def test_rename_results_output_already_in_replacements(self):
        # Output 'z' is already in replacements with the same value (final output)
        nodes = [oh.make_node("Relu", ["x"], ["z"], name="n0")]
        replacements = {"x": "x", "z": "z"}
        new_nodes = self.native_alias_cleanup(nodes, replacements)
        self.assertEqual(list(new_nodes[0].output), ["z"])

    def test_rename_results_multiple_nodes(self):
        # Add(x, y) -> tmp; Relu(tmp) -> out
        nodes = [
            oh.make_node("Add", ["x", "y"], ["tmp"], name="n0"),
            oh.make_node("Relu", ["tmp"], ["out"], name="n1"),
        ]
        replacements = {"x": "new_x", "y": "new_y", "out": "out"}
        new_nodes = self.native_alias_cleanup(nodes, replacements)
        self.assertEqual(len(new_nodes), 2)
        # Inputs of first node are renamed
        self.assertEqual(list(new_nodes[0].input), ["new_x", "new_y"])
        # Output of first node becomes the input of the second node
        first_out = new_nodes[0].output[0]
        self.assertEqual(list(new_nodes[1].input), [first_out])
        # Final output is preserved
        self.assertEqual(list(new_nodes[1].output), ["out"])

    def test_rename_results_with_graph_attribute(self):
        # Build a minimal If node with a then_branch subgraph
        then_graph = oh.make_graph(
            [oh.make_node("Add", ["outer_x", "outer_y"], ["branch_out"])],
            "then_branch",
            [],
            [oh.make_tensor_value_info("branch_out", TensorProto.FLOAT, [])],
        )
        if_node = oh.make_node("If", ["cond"], ["result"], name="n0")
        if_node.attribute.append(oh.make_attribute("then_branch", then_graph))
        if_node.attribute.append(oh.make_attribute("else_branch", then_graph))
        replacements = {
            "cond": "cond",
            "result": "result",
            "outer_x": "outer_x",
            "outer_y": "outer_y",
        }
        new_nodes = self.native_alias_cleanup([if_node], replacements)
        self.assertEqual(len(new_nodes), 1)
        self.assertEqual(new_nodes[0].op_type, "If")

    def test_rename_results_in_subgraph_no_replacement_needed(self):
        subgraph = oh.make_graph(
            [oh.make_node("Add", ["a", "b"], ["c"])],
            "sub",
            [],
            [oh.make_tensor_value_info("c", TensorProto.FLOAT, [])],
        )
        # No actual substitution: replacements map each name to itself
        replacements = {"a": "a", "b": "b"}
        result = self.native_subgraph_alias_cleanup(subgraph, replacements)
        self.assertEqual(list(result.node[0].input), ["a", "b"])
        self.assertEqual(list(result.node[0].output), ["c"])

    def test_rename_results_in_subgraph_with_replacement(self):
        subgraph = oh.make_graph(
            [oh.make_node("Add", ["a", "b"], ["c"])],
            "sub",
            [],
            [oh.make_tensor_value_info("c", TensorProto.FLOAT, [])],
        )
        replacements = {"a": "new_a", "b": "b"}
        result = self.native_subgraph_alias_cleanup(subgraph, replacements)
        self.assertTrue(str(result.name).startswith("sub"))
        self.assertEqual(len(result.node), 1)
        self.assertEqual(list(result.node[0].input), ["new_a", "b"])
        self.assertEqual(list(result.node[0].output), ["c"])

    def test_rename_results_in_subgraph_shadowing(self):
        """Rejects a subgraph that redefines an ancestor's value."""
        subgraph = oh.make_graph(
            [
                oh.make_node("Add", ["a", "b"], ["a"]),  # shadows 'a'
                oh.make_node("Relu", ["a"], ["c"]),
            ],
            "sub",
            [],
            [oh.make_tensor_value_info("c", TensorProto.FLOAT, [])],
        )
        replacements = {"a": "new_a", "b": "b"}
        with self.assertRaisesRegex(ValueError, "'a'.*SSA shadowing is not allowed"):
            self.native_subgraph_alias_cleanup(subgraph, replacements)

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
        g2 = g.empty_copy(as_function=True)
        self.assertIsInstance(g2, GraphBuilder)
        self.assertTrue(g2.shapes_context.empty())
        g2.make_tensor_input("X", TFLOAT, (2, 3))
        g2.op.Relu("X", outputs=["Y"])
        self.assertEqual(g2.get_shape("Y"), (2, 3))
        self.assertFalse(g.has_name("X"))

    def test_pretty_tensor_with_shape(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = onh.from_array(arr)
        result = str(tensor)
        self.assertEqual(tensor.data_type, TFLOAT)
        self.assertIn("2", result)

    def test_pretty_tensor_without_shape(self):
        tensor = onh.from_array(np.array(5, dtype=np.int64))
        self.assertEqual(list(tensor.dims), [])
        self.assertIn("data_type", str(tensor))

    def test_pretty_node_shape_op(self):
        node = oh.make_node("Shape", ["X"], ["shape_out"])
        result = str(node)
        self.assertIn("Shape", result)
        self.assertIn("X", result)
        self.assertIn("shape_out", result)

    def test_pretty_node_shape_op_with_attributes(self):
        node = oh.make_node("Shape", ["X"], ["shape_out"], start=1, end=3)
        result = str(node)
        self.assertIn("Shape", result)
        self.assertEqual({str(a.name): a.i for a in node.attribute}, {"start": 1, "end": 3})

    def test_pretty_node_shape_true(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        node = oh.make_node("Add", ["X", "Y"], ["Z"])
        g.set_type("X", TFLOAT)
        g.set_shape("X", (2, 3))
        g.set_type("Y", TFLOAT)
        g.set_shape("Y", (2, 3))
        g.set_type("Z", TFLOAT)
        g.set_shape("Z", (2, 3))
        self.assertIn("Add", str(node))
        for name in ("X", "Y", "Z"):
            self.assertEqual(g.shapes_context.get(name).dtype, TFLOAT)
            self.assertEqual(g.shapes_context.get(name).shape, SymShape([2, 3]))

    def test_pretty_node_shape_op_with_shape_true(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        node = oh.make_node("Shape", ["X"], ["shape_out"])
        g.set_type("X", TFLOAT)
        g.set_shape("X", (3, 4))
        g.set_type("shape_out", TINT64)
        g.set_shape("shape_out", (2,))
        result = str(node)
        self.assertIn("Shape", result)
        self.assertEqual(g.shapes_context.get("X").shape, SymShape([3, 4]))
        self.assertEqual(g.shapes_context.get("shape_out").shape, SymShape([2]))
        self.assertEqual(g.get_type("shape_out"), TINT64)

    def test_do_not_turn_constant_initializers_flag_set(self):
        # Explicit lowering creates a Constant without retaining initializer storage.
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", TFLOAT, (2, 4), False)
        np_weights = np.ones((4, 3), dtype=np.float32)
        g.make_initializer("weights", np_weights)
        g.move_initializers_to_constant(full_parameter_name=False)
        self.assertNotIn("weights", g.initializers_dict)
        self.assertEqual([str(node.op_type) for node in g.nodes], ["Constant"])
        self.assertTrue(g.constant_is_equal_to("weights", np_weights))

    def test_do_not_turn_constant_initializers_no_parent(self):
        # Initializers retain native ownership until explicitly lowered.
        g = GraphBuilder(18, ir_version=9)
        g.make_initializer("cst", np.ones((2, 2), dtype=np.float32))
        self.assertIn("cst", g.initializers_dict)
        self.assertFalse(g.is_constant("other"))

    def test_do_not_turn_constant_initializers_parent_does_not_have_name(self):
        # Parent does not know the name at all: returns False.
        parent = GraphBuilder(18, ir_version=9)
        child = parent.empty_copy()
        child.make_initializer("cst", np.arange(4).reshape((2, 2)).astype(np.float32))
        child.move_initializers_to_constant()
        self.assertTrue(child.is_constant("cst"))
        self.assertFalse(parent.is_constant("cst"))

    def test_do_not_turn_constant_initializers_same_constant_in_parent(self):
        # Same constant in both parent and child: has_exact_same_constant_in_context returns True,
        # so the method returns not True = False (safe to share with parent).
        parent = GraphBuilder(18, ir_version=9)
        np_cst = np.arange(4).reshape((2, 2)).astype(np.float32)
        parent.make_initializer("cst", np_cst)

        child = parent.empty_copy()
        child.make_initializer("cst", np_cst)

        child.move_initializers_to_constant()
        self.assertNotIn("cst", child.initializers_dict)
        self.assertIn("cst", parent.initializers_dict)
        self.assertTrue(child.constant_is_equal_to("cst", parent.get_constant("cst")))

    def test_do_not_turn_constant_initializers_different_constant_in_parent(self):
        # Different values for same name: has_exact_same_constant_in_context returns False,
        # so the method returns not False = True (shadowing would occur).
        parent = GraphBuilder(18, ir_version=9)
        np_cst1 = np.arange(4).reshape((2, 2)).astype(np.float32)
        parent.make_initializer("cst", np_cst1)

        child = parent.empty_copy()
        np_cst2 = np_cst1 + 1.0
        child.make_initializer("cst", np_cst2)

        child.move_initializers_to_constant()
        self.assertTrue(child.constant_is_equal_to("cst", np_cst2))
        self.assertTrue(parent.constant_is_equal_to("cst", np_cst1))

    def test_do_not_turn_constant_initializers_large_constant_recurse_to_parent(self):
        # Large constants (>= 128 elements) cause has_exact_same_constant_in_context to return
        # None, so the method recurses to the parent. The parent has no parent, so it returns
        # False.
        parent = GraphBuilder(18, ir_version=9)
        np_cst = np.arange(128).reshape((16, 8)).astype(np.float32)
        parent.make_initializer("cst", np_cst)

        child = parent.empty_copy()
        child.make_initializer("cst", np_cst)

        child.move_initializers_to_constant()
        self.assertTrue(child.constant_is_equal_to("cst", parent.get_constant("cst")))
        self.assertEqual(len(child.nodes), 1)
        self.assertEqual(len(parent.nodes), 0)


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
        msg = gr.pretty_text()
        self.assertIsInstance(msg, str)
        self.assertIn("Abs", msg)
        self.assertIn("Neg", msg)
        self.assertIn("Relu", msg)
        self.assertEqual([str(n.name) for n in gr.nodes], ["n0", "n1", "n2"])

    def test_position_msg_single_node(self):
        model = self._make_simple_model()
        gr = GraphBuilder(model)
        node = gr.nodes[1]  # Neg node
        msg = str(node)
        self.assertIsInstance(msg, str)
        self.assertIn("Neg", msg)
        self.assertEqual(list(node.input), ["a"])
        self.assertEqual(list(node.output), ["b"])

    def test_position_msg_with_around(self):
        model = self._make_simple_model()
        gr = GraphBuilder(model)
        msg = gr.pretty_text()
        self.assertIsInstance(msg, str)
        self.assertIn("Neg", msg)
        self.assertLess(msg.index("Abs"), msg.index("Neg"))
        self.assertLess(msg.index("Neg"), msg.index("Relu"))

    def test_position_msg_with_none_node(self):
        model = self._make_simple_model()
        gr = GraphBuilder(model)
        msg = gr.pretty_text()
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
        g.make_tensor_output("output_0", TFLOAT, (2, 3), indexed=False)
        onx = g.to_onnx(optimize=False)
        vi_names = {vi.name for vi in onx.graph.value_info}
        self.assertIn("tmp", vi_names, "intermediate tensor 'tmp' must have shape info")
        tmp_vi = next(vi for vi in onx.graph.value_info if vi.name == "tmp")
        shape = [d.dim_value for d in tmp_vi.type.tensor_type.shape.dim]
        self.assertEqual(shape, [2, 3])

    def test_value_info_dynamic_shapes(self):
        """Shape info for intermediate tensors must also be present with dynamic dims."""
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, ("batch", 3))
        g.make_node("Relu", ["X"], ["tmp"], name="relu1")
        g.make_node("Relu", ["tmp"], ["output_0"], name="relu2")
        g.make_tensor_output("output_0", TFLOAT, ("batch", 3), indexed=False)
        onx = g.to_onnx(optimize=False)
        vi_names = {vi.name for vi in onx.graph.value_info}
        self.assertIn("tmp", vi_names, "intermediate tensor 'tmp' must have shape info")
        tmp_vi = next(vi for vi in onx.graph.value_info if vi.name == "tmp")
        shape = [d.dim_param or d.dim_value for d in tmp_vi.type.tensor_type.shape.dim]
        self.assertEqual(shape, ["batch", 3])

    def test_optimize_applies_dynamic_dimension_renaming(self):
        """Preserves explicit symbolic annotations through native optimization."""
        g = GraphBuilder(18, ir_version=9, optimization_options=OptimizationOptions(patterns=[]))
        g.make_tensor_input("X", TFLOAT, ("s0", "s1"))
        g.set_shape("X", ("batch", "seq"))
        g.make_node("Relu", ["X"], ["Y"], name="A")
        g.make_tensor_output("Y")
        model = g.to_onnx()
        self.assertEqual(g.get_shape("X"), ("batch", "seq"))
        self.assertEqual(g.get_shape("Y"), ("batch", "seq"))
        for value in (*model.graph.input, *model.graph.output):
            self.assertEqual(
                [str(d.dim_param) for d in value.type.tensor_type.shape.dim], ["batch", "seq"]
            )

    def test_no_duplicate_batch_names_multiple_outputs(self):
        """Preserves a shared symbolic batch dimension on every native output."""
        g = GraphBuilder(18, ir_version=9)
        g.make_tensor_input("X", TFLOAT, ("batch", 4))
        g.make_node("Relu", ["X"], ["Y0"], name="relu0")
        g.make_node("Relu", ["X"], ["Y1"], name="relu1")
        g.make_node("Relu", ["X"], ["Y2"], name="relu2")
        for out_name in ("Y0", "Y1", "Y2"):
            g.make_tensor_output(out_name)
            self.assertEqual(g.get_shape(out_name), ("batch", 4))
        model = g.to_onnx()
        self.assertEqual(len(model.graph.output), 3)
        for value in model.graph.output:
            self.assertEqual(str(value.type.tensor_type.shape.dim[0].dim_param), "batch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
