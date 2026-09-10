"""Tests the onnx-light graph builder adapter used by Torch conversion."""

import unittest

import numpy
from onnx_light import onnx

from yobx.torch.interpreter.graph_builder import TorchOnnxLightGraphBuilder
from yobx.xbuilder._wrap_dim import WrapDim
from yobx.xbuilder.function_options import FunctionOptions


class TestTorchOnnxLightGraphBuilder(unittest.TestCase):
    """Tests Torch-specific helpers without running a full model export."""

    def test_constructor_and_child_builder_preserve_adapter(self):
        """Accepts legacy export state and preserves the subclass in children."""
        builder = TorchOnnxLightGraphBuilder(
            18,
            input_names=["X"],
            args=(1,),
            kwargs={"flag": True},
            dynamic_shapes=({0: "batch"},),
            local_domain="torch.local",
            output_names=["Y"],
            output_dynamic_shapes=({0: "batch"},),
            signature="signature",
            check_empty_source=True,
            graph_module="graph",
            exe_path="test",
        )
        self.assertEqual(builder.input_args, (1,))
        self.assertEqual(builder.input_kwargs, {"flag": True})
        self.assertIn("batch", builder.dynamic_objects)
        child = builder.empty_copy(as_function=True)
        self.assertIsInstance(child, TorchOnnxLightGraphBuilder)
        self.assertTrue(child.as_function)
        self.assertEqual(child.local_domain, "torch.local")

    def test_naming_rank_type_stats_and_users(self):
        """Provides Torch naming, rank, dtype, statistics, and user bookkeeping."""
        import torch

        builder = TorchOnnxLightGraphBuilder(18)
        builder.make_tensor_input("X", onnx.TensorProto.FLOAT, (2, 3))
        self.assertEqual(builder.rank("X"), 2)
        node_name = builder.unique_node_name("node")
        self.assertEqual(node_name, "node")
        self.assertEqual(builder.unique_node_name("node"), "node2")
        builder.op.Identity("X", outputs=["identity"], name=node_name)
        self.assertEqual(str(builder.nodes[-1].name), "node")
        builder.set_shapes_types("Y", "run_node", ("", ("Y", torch.float16, torch.Size([2, 3]))))
        self.assertEqual(builder.get_type_known("Y"), onnx.TensorProto.FLOAT16)
        builder.add_stat("aten", "relu")
        builder.add_stat("aten", "relu")
        self.assertEqual(builder.statistics_, {"aten": {"relu": 2}})
        builder.register_users("X", ["relu", "add"])
        self.assertEqual(builder._registered_users["X"], {"relu", "add"})
        with self.assertRaises(AssertionError):
            builder.register_users("X", [])

    def test_make_shape_from_results_static_and_dynamic(self):
        """Builds cached static and dynamic runtime shape tensors."""
        builder = TorchOnnxLightGraphBuilder(18, dynamic_shapes=({0: "batch"},))
        builder.make_tensor_input("X", onnx.TensorProto.FLOAT, ("batch", 3))

        static = builder.make_shape_from_results([2, 3])
        numpy.testing.assert_array_equal(
            builder.get_constant(static), numpy.array([2, 3], dtype=numpy.int64)
        )
        self.assertEqual(static, builder.make_shape_from_results([2, 3]))

        dynamic = builder.make_shape_from_results(["batch", 3], name="reshape")
        self.assertEqual(builder.get_type(dynamic), onnx.TensorProto.INT64)
        self.assertEqual(builder.get_shape(dynamic), (2,))
        self.assertEqual(builder.value_as_shape(dynamic), ("batch", 3))
        self.assertEqual(dynamic, builder.make_shape_from_results(["batch", 3]))
        self.assertIn("Gather", [str(node.op_type) for node in builder.nodes])
        self.assertIn("Concat", [str(node.op_type) for node in builder.nodes])

    def test_dynamic_shape_and_compatibility_helpers(self):
        """Normalizes dynamic specifications and compares symbolic shapes."""
        builder = TorchOnnxLightGraphBuilder(18, dynamic_shapes={"X": {0: "batch"}})
        shape = builder.get_input_dynamic_shape("X", 0, (2, 3))
        self.assertEqual(shape, ("batch", 3))
        self.assertTrue(builder.is_dynamic_shape(shape))
        self.assertFalse(builder.is_dynamic_shape((2, 3)))
        builder.make_tensor_input("X", onnx.TensorProto.FLOAT, shape)
        builder.make_tensor_input("Y", onnx.TensorProto.FLOAT, ("other", 3))
        self.assertFalse(builder.same_shape("X", "Y"))
        builder._check_two_shapes_are_compatible(
            builder.get_shape("X"), builder.get_shape("Y"), name="XY"
        )
        self.assertTrue(builder.same_shape("X", "Y"))
        with self.assertRaises(AssertionError):
            builder._check_two_shapes_are_compatible((2, 3), (2, 4), name="bad")

    def test_make_new_dynamic_shape_uses_valid_wrappers(self):
        """Creates fresh dimensions without constructing invalid Torch SymInt values."""
        builder = TorchOnnxLightGraphBuilder(18)
        shape = builder.make_new_dynamic_shape(2, prefix="fresh")
        self.assertTrue(all(isinstance(dimension, WrapDim) for dimension in shape))
        self.assertEqual(builder.verify_dynamic_shape(shape), ("fresh_d0", "fresh_d1"))
        self.assertIn("fresh_d0", builder.dynamic_objects)
        self.assertIn("fresh_d1", builder.dynamic_objects)

    def test_sequence_input_and_local_function_helpers(self):
        """Declares sequence inputs and exposes native local-function metadata."""
        builder = TorchOnnxLightGraphBuilder(18, input_names=["renamed_sequence"])
        builder.make_tensor_sequence_input("sequence", onnx.TensorProto.FLOAT, (2, 3))
        self.assertEqual(builder.current_input, 1)
        self.assertEqual(builder.input_names, ["renamed_sequence"])
        self.assertTrue(builder.is_sequence("renamed_sequence"))
        self.assertTrue(builder.is_sequence("sequence"))
        self.assertEqual(
            builder.get_sequence("sequence"),
            {
                "dtype": onnx.TensorProto.FLOAT,
                "shapes": ((2, 3),),
                "ranks": (2,),
                "unknown": False,
            },
        )
        self.assertTrue(builder.inputs[0].type.HasField("sequence_type"), builder.inputs[0])
        position = builder.make_initializer("", numpy.array(0, dtype=numpy.int64))
        item = builder.make_node("SequenceAt", ["sequence", position], ["item"])
        self.assertEqual(item, "item")
        self.assertEqual(builder.get_type(item), onnx.TensorProto.FLOAT)
        self.assertEqual(builder.get_shape(item), (2, 3))
        builder.make_tensor_output(item)
        onnx.checker.check_model(builder.to_native(optimize=False))

        child = TorchOnnxLightGraphBuilder(18, as_function=True)
        child.make_tensor_input("x", onnx.TensorProto.FLOAT, (2, 3))
        child.op.Neg("x", outputs=["y"])
        child.make_tensor_output("y")
        parent = TorchOnnxLightGraphBuilder(18)
        parent.make_tensor_input("X", onnx.TensorProto.FLOAT, (2, 3))
        options = FunctionOptions(
            name="neg",
            domain="torch.local",
            export_as_function=True,
            inline=False,
            move_initializer_to_constant=True,
            return_initializer=True,
        )
        self.assertEqual(
            parent.make_nodes(child, ["X"], ["Y"], function_options=options, optimize=False), "Y"
        )
        self.assertTrue(parent.has_local_function("neg", "torch.local"))
        self.assertTrue(parent.has_local_function("neg", "torch.local", builder=True))
        self.assertEqual(parent.get_local_function_outputs("neg", "torch.local"), ("y",))
        self.assertEqual(parent.get_shape("Y"), (2, 3))

    def test_process_orchestration(self):
        """Runs graph lifecycle hooks and skips trailing unused scalar inputs."""

        class Node:
            def __init__(self, name, op, target, users=None, value=None):
                self.name = name
                self.op = op
                self.target = target
                self.users = users or {}
                self.meta = {} if value is None else {"val": value}

        class Graph:
            def __init__(self, nodes):
                self.nodes = nodes

        class Module:
            def __init__(self, graph):
                self.graph = graph

        class Interpreter:
            def __init__(self):
                self.events = []

            def start_graph(self, graph):
                self.events.append(("start", graph))

            def run_node(self, node, source_lines=None):
                self.events.append(("node", node.name, source_lines))

            def end_graph(self, graph):
                self.events.append(("end", graph))

        used = Node("X", "placeholder", "X", users={"neg": None}, value=object())
        unused = Node("flag", "placeholder", "flag", value=True)
        output = Node("output", "output", "output", value=object())
        graph = Graph([used, unused, output])
        interpreter = Interpreter()
        builder = TorchOnnxLightGraphBuilder(18)
        builder.process(Module(graph), interpreter, source_lines={"source": ()})
        self.assertEqual(
            [event[1] for event in interpreter.events if event[0] == "node"], ["X", "output"]
        )
        self.assertEqual(interpreter.events[0], ("start", graph))
        self.assertEqual(interpreter.events[-1], ("end", graph))

    def test_legacy_fx_interpreter_uses_adapter(self):
        """Runs a simple legacy FX lowering with the dedicated adapter."""
        import torch

        from yobx.torch.export_options import ExportOptions
        from yobx.torch.interpreter.interpreter import FxGraphInterpreter, GraphBuilder

        class Model(torch.nn.Module):
            def forward(self, x):
                """Applies a ReLU operation."""
                return torch.relu(x)

        self.assertIs(GraphBuilder, TorchOnnxLightGraphBuilder)
        graph_module = torch.fx.symbolic_trace(Model())
        example = torch.randn(2, 3)
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = example
            elif node.op == "call_function":
                node.meta["val"] = torch.relu(example)
        builder = TorchOnnxLightGraphBuilder(18, args=(example,), graph_module=graph_module)
        interpreter = FxGraphInterpreter(
            builder,
            lambda *args, **kwargs: None,
            example_inputs=(example,),
            export_options=ExportOptions(),
        )
        builder.process(graph_module, interpreter)
        self.assertEqual(builder.output_names, ["output"])
        self.assertEqual([str(node.op_type) for node in builder.nodes], ["Relu", "Identity"])

    def test_legacy_export_factory_uses_adapter(self):
        """Constructs the dedicated adapter in the legacy export setup."""
        import torch

        from yobx.torch.export_options import ExportOptions
        from yobx.torch.interpreter.onnx_export import _make_builder_interpreter

        class Model(torch.nn.Module):
            def forward(self, x):
                """Returns one tensor operation."""
                return torch.relu(x)

        example = torch.randn(2, 3)
        _, builder, _, _ = _make_builder_interpreter(
            Model(), args=(example,), export_options=ExportOptions(), target_opset=18
        )
        self.assertIsInstance(builder, TorchOnnxLightGraphBuilder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
