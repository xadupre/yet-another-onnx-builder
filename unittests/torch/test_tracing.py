import unittest
import torch
from yobx.ext_test_case import ExtTestCase, requires_torch


@requires_torch("2.0")
class TestCustomTracer(ExtTestCase):
    def test_import(self):
        from yobx.torch.tracing import (
            CustomTracer,
            CustomProxy,
            CustomAttribute,
            CustomParameterProxy,
            CustomProxyInt,
            CustomProxyFloat,
            CondCCOp,
            LEAVE_INPLACE,
            _len,
            _isinstance,
            replace_problematic_function_before_tracing,
            setitem_with_transformation,
            tree_unflatten_with_proxy,
        )
        self.assertIsNotNone(CustomTracer)
        self.assertIsNotNone(CustomProxy)

    def test_import_from_package(self):
        from yobx.torch import CustomTracer, CustomProxy

        self.assertIsNotNone(CustomTracer)
        self.assertIsNotNone(CustomProxy)

    def test_trace_simple_add(self):
        from yobx.torch.tracing import CustomTracer

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        graph = CustomTracer().trace(Model())
        ops = [n.op for n in graph.nodes]
        self.assertIn("placeholder", ops)
        self.assertIn("output", ops)

    def test_trace_linear(self):
        from yobx.torch.tracing import CustomTracer

        model = torch.nn.Linear(4, 4)
        graph = CustomTracer().trace(model)
        self.assertIsNotNone(graph)
        node_ops = {n.op for n in graph.nodes}
        self.assertIn("placeholder", node_ops)
        self.assertIn("output", node_ops)

    def test_trace_inplace_add(self):
        from yobx.torch.tracing import CustomTracer

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.clone()
                y += 1
                return y

        graph = CustomTracer().trace(Model())
        # The inplace add_ should be removed/replaced
        self.assertIsNotNone(graph)
        graph.lint()

    def test_custom_proxy_repr(self):
        from yobx.torch.tracing import CustomTracer, CustomProxy

        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        tracer = CustomTracer()
        graph = tracer.trace(Model())
        # Check that placeholders are CustomProxy instances
        for node in graph.nodes:
            if node.op == "placeholder":
                proxy = tracer.proxy(node)
                self.assertIsInstance(proxy, CustomProxy)
                self.assertIn("CustomProxy", repr(proxy))

    def test_is_leaf_module_default(self):
        from yobx.torch.tracing import CustomTracer

        tracer = CustomTracer()
        # Standard nn.Linear is a leaf by default
        linear = torch.nn.Linear(4, 4)
        self.assertTrue(tracer.is_leaf_module(linear, "linear"))

    def test_is_leaf_module_custom(self):
        from yobx.torch.tracing import CustomTracer

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x

        tracer = CustomTracer(
            module_leaves={MyModule: lambda m, module_qualified_name: True}
        )
        self.assertTrue(tracer.is_leaf_module(MyModule(), "mymodule"))

    def test_len_with_proxy(self):
        from yobx.torch.tracing import CustomTracer, _len

        class Model(torch.nn.Module):
            def forward(self, x):
                return x

        tracer = CustomTracer()
        graph = tracer.trace(Model())
        for node in graph.nodes:
            if node.op == "placeholder":
                proxy = tracer.proxy(node)
                # _len on a proxy should return another proxy
                result = _len(proxy)
                self.assertIsNotNone(result)
                break

    def test_len_plain(self):
        from yobx.torch.tracing import _len

        self.assertEqual(_len([1, 2, 3]), 3)

    def test_isinstance_plain(self):
        from yobx.torch.tracing import _isinstance

        self.assertTrue(_isinstance([1, 2], list))
        self.assertFalse(_isinstance((1, 2), list))

    def test_replace_problematic_function(self):
        from yobx.torch.tracing import replace_problematic_function_before_tracing

        original_cat = torch.cat
        with replace_problematic_function_before_tracing():
            # Inside context: torch.cat is replaced
            self.assertIsNot(torch.cat, original_cat)
        # After context: torch.cat is restored
        self.assertIs(torch.cat, original_cat)

    def test_cond_cc_op(self):
        from yobx.torch.tracing import CondCCOp

        op = CondCCOp()
        self.assertIsNotNone(op)
        self.assertIsInstance(op, torch._ops.HigherOrderOperator)

    def test_trace_with_module_leaves(self):
        from yobx.torch.tracing import CustomTracer

        class LeafModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = LeafModule()

            def forward(self, x):
                return self.leaf(x) + 1

        tracer = CustomTracer(
            module_leaves={LeafModule: lambda m, module_qualified_name: True}
        )
        graph = tracer.trace(OuterModule())
        # LeafModule should appear as call_module node
        module_calls = [n for n in graph.nodes if n.op == "call_module"]
        self.assertEqual(len(module_calls), 1)

    def test_create_arg_types(self):
        from yobx.torch.tracing import CustomTracer

        tracer = CustomTracer()
        # Need a root module for context
        model = torch.nn.Linear(4, 4)
        _ = tracer.trace(model)
        self.assertEqual(tracer.create_arg(bool), torch.bool)
        self.assertEqual(tracer.create_arg(int), torch.int64)
        self.assertEqual(tracer.create_arg(float), torch.float32)
        self.assertEqual(tracer.create_arg(complex), torch.complex64)

    def test_remove_inplace_no_inplace(self):
        from yobx.torch.tracing import CustomTracer

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        tracer = CustomTracer()
        graph = tracer.trace(Model())
        # No inplace to remove
        result = CustomTracer.remove_inplace(graph)
        self.assertEqual(result, 0)

    def test_trace_setitem(self):
        from yobx.torch.tracing import CustomTracer

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.clone()
                y[0] = 0
                return y

        graph = CustomTracer().trace(Model())
        self.assertIsNotNone(graph)
        graph.lint()


if __name__ == "__main__":
    unittest.main(verbosity=2)
