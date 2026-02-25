import itertools
import unittest

import torch
from onnx_pipe.tracing.mixed_export_builder import MixedExportTracer
from torch.fx._lazy_graph_module import _make_graph_module
from torch.export.graph_signature import (
    InputSpec,
    OutputSpec,
    TensorArgument,
    InputKind,
    OutputKind,
    ExportGraphSignature,
)


class TestMixedExportBuilder(unittest.TestCase):
    def test_tracing_submodule(self):
        class SubModule(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1, kind: int = 0):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))
                self.kind = kind

            def forward(self, x):
                return torch.sigmoid(self.linear(x)) + self.buff

        class Model(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super().__init__()
                self.suba = SubModule(n_dims, n_targets, kind=1)
                self.subb = SubModule(n_dims, n_targets, kind=2)

            def forward(self, x, y):
                return self.suba(x) + self.subb(y)

        def f(mod, module_qualified_name=None):
            self.assertIsInstance(module_qualified_name, str)
            self.assertIn(module_qualified_name, ("suba", "subb"))
            return mod.kind == 1

        module_leaves = {SubModule: f}
        model = Model()
        self.assertTrue(f(model.suba, "suba"))
        self.assertFalse(f(model.subb, "suba"))
        graph = MixedExportTracer(module_leaves=module_leaves).trace(model)
        module_nodes = [n for n in graph.nodes if n.op == "call_module"]
        self.assertEqual(len(module_nodes), 2)
        self.assertEqual(module_nodes[0].target, "suba")
        self.assertEqual(module_nodes[1].target, "subb.linear")

    def test_tracing_submodule_with_test(self):
        class SubModule(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1, kind: int = 0):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))
                self.kind = kind

            def forward(self, x):
                return torch.sigmoid(self.linear(x)) + self.buff

        class SubModuleDoNotTrace(torch.nn.Module):
            def forward(self, x):
                if x.sum().item() > 0:
                    return x
                return -x

        class Model(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super().__init__()
                self.suba = SubModule(n_dims, n_targets, kind=1)
                self.subdonot = SubModuleDoNotTrace()

            def forward(self, x, y):
                return self.suba(x) + self.subdonot(y)

        def f(mod, module_qualified_name=None):
            return isinstance(mod, SubModuleDoNotTrace)

        module_leaves = {SubModuleDoNotTrace: f}
        model = Model()
        graph = MixedExportTracer(module_leaves=module_leaves).trace(model)
        module_nodes = [n for n in graph.nodes if n.op == "call_module"]
        self.assertEqual(len(module_nodes), 2)
        self.assertEqual(module_nodes[0].target, "suba.linear")
        self.assertEqual(module_nodes[1].target, "subdonot")

    def test_tracing_submodule_concrete_args(self):
        class SubModule(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1, kind: int = 0):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))
                self.kind = kind

            def forward(self, x):
                return torch.sigmoid(self.linear(x)) + self.buff

        class Model(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super().__init__()
                self.suba = SubModule(n_dims, n_targets, kind=1)
                self.subb = SubModule(n_dims, n_targets, kind=2)

            def forward(self, x, y):
                return self.suba(x) + self.subb(y)

        def f(mod, module_qualified_name=None):
            self.assertIsInstance(module_qualified_name, str)
            self.assertIn(module_qualified_name, ("suba", "subb"))
            return mod.kind == 1

        module_leaves = {SubModule: f}
        model = Model()
        self.assertTrue(f(model.suba, "suba"))
        self.assertFalse(f(model.subb, "suba"))
        kwargs = dict(x=torch.randn((5, 3)), y=torch.randn((5, 3)))
        dynamic_shapes = dict(x={0: "batch"}, y={0: "batch"})
        tracer = MixedExportTracer(module_leaves=module_leaves)
        graph = tracer.trace(model, kwargs, dynamic_shapes=dynamic_shapes)

        module_nodes = [n for n in graph.nodes if n.op == "call_module"]
        self.assertEqual(len(module_nodes), 2)
        self.assertEqual(module_nodes[0].target, "suba")
        self.assertEqual(module_nodes[1].target, "subb.linear")
        gm = _make_graph_module(tracer.root, graph, model.__class__.__name__)

        input_specs = [
            InputSpec(
                kind=InputKind.USER_INPUT, arg=TensorArgument(name="x"), target=None
            ),
            InputSpec(
                kind=InputKind.USER_INPUT, arg=TensorArgument(name="y"), target=None
            ),
        ]
        output_specs = [
            OutputSpec(
                kind=OutputKind.USER_OUTPUT,
                arg=TensorArgument(name="add_1"),
                target=None,
            )
        ]
        torch.export._trace._EXPORT_MODULE_HIERARCHY = (
            torch.export._trace._get_module_hierarchy(model)
        )

        class MyExportedProgram(torch.export.ExportedProgram):
            def validate(self):
                pass

        ep = MyExportedProgram(
            root=gm,
            graph=gm.graph,
            graph_signature=ExportGraphSignature(
                input_specs=input_specs, output_specs=output_specs
            ),
            # call_spec=call_spec,
            state_dict=gm.state_dict(),
            range_constraints={},
            example_inputs=((), kwargs),
            module_call_graph=torch.export._trace._make_module_call_graph(
                input_specs, output_specs, {}
            ),
        )
        self.assertIn("suba = self.suba(x);  x = None", str(ep))
        self.assertNotIn("f32[5, 3]", str(ep))
        """
        print(ep)
        ExportedProgram:
            class GraphModule(torch.nn.Module):
                def forward(self, x: "f32[s26, 3]", y: "f32[s26, 3]"):
                    # No stacktrace found for following nodes
                    suba = self.suba(x);  x = None
                    subb_linear = self.subb.linear(y);  y = None
                    sigmoid = torch.sigmoid(subb_linear);  subb_linear = None
                    subb_buff = self.subb.buff
                    add = sigmoid + subb_buff;  sigmoid = subb_buff = None
                    add_1 = suba + add;  suba = add = None
                    return add_1
            
        Graph signature: 
            # inputs
            x: USER_INPUT
            y: USER_INPUT
            
            # outputs
            add_1: USER_OUTPUT
            
        Range constraints: {}        
        """
        # fails with decomposition
        # epo = torch.onnx.export(ep, args=(), kwargs=kwargs, dynamic_shapes=dynamic_shapes)
        # print(epo)

    def test_io_captured_custom_class(self):
        class TestCustomClass:
            def __init__(self, keys, values):
                self.data = list(zip(keys, values))

        def _flatten(custom):
            data = custom.data
            flat = list(itertools.chain.from_iterable(data))
            keys = list(
                itertools.chain.from_iterable(
                    (f"key_{i}", f"value_{i}") for i in range(len(data))
                )
            )
            return flat, keys

        def _flatten_with_keys(custom):
            values, context = _flatten(custom)
            return [
                (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
            ], context

        def _unflatten(values, context, output_type=None):
            return TestCustomClass(values[::2], values[1::2])

        torch.utils._pytree.register_pytree_node(
            TestCustomClass,
            _flatten,
            _unflatten,
            serialized_type_name="onnxtest.TestCustomClass",
            flatten_with_keys_fn=_flatten_with_keys,
        )

        class Model(torch.nn.Module):
            def forward(self, x, custom=None):
                if not custom:
                    return x
                data = custom.data
                return x + data[0][0] + data[0][1] + data[1][0] + data[1][1]

        kwargs = dict(
            x=torch.randn((6, 7)),
            custom=TestCustomClass(
                [torch.randn((6, 7)), torch.randn((1, 7))],
                [torch.randn((1, 7)), torch.randn((6, 7))],
            ),
        )

        model = Model()
        model(**kwargs)

        dynamic_shapes = dict(
            x={0: "batch", 1: "seq"},
            custom=[
                {0: "batch", 1: "seq"},
                {1: "seq"},
                {1: "seq"},
                {0: "batch", 1: "seq"},
            ],
        )
        tracer = MixedExportTracer()
        with self.assertRaises(NotImplementedError):
            graph = tracer.trace(model, kwargs, dynamic_shapes=dynamic_shapes)
            print(graph)


if __name__ == "__main__":
    unittest.main()
