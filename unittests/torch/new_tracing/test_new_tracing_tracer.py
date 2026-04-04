"""Tests for yobx.torch.new_tracing – tracing (GraphTracer and trace_model)."""

import operator
import unittest
import torch
from yobx.ext_test_case import ExtTestCase
from yobx.torch.new_tracing.shape import TracingInt
from yobx.torch.new_tracing.tracer import GraphTracer
from yobx.torch.new_tracing import trace_model


class TestNewTracingTracer(ExtTestCase):
    def test_trace_simple_add(self):
        def add(x, y):
            return x + y

        tracer = GraphTracer()
        graph = tracer.trace(add, (torch.randn(3, 4), torch.randn(3, 4)))

        ops = [n.op for n in graph.nodes]
        self.assertIn("placeholder", ops)
        self.assertIn("call_function", ops)
        self.assertIn("output", ops)
        # Two placeholders for the two inputs.
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 2)
        graph.lint()

    def test_trace_simple_mul(self):
        def mul(x, y):
            return x * y

        tracer = GraphTracer()
        graph = tracer.trace(mul, (torch.randn(2, 5), torch.randn(2, 5)))
        graph.lint()
        call_nodes = [n for n in graph.nodes if n.op == "call_function"]
        self.assertGreater(len(call_nodes), 0)

    def test_trace_elementwise_chain(self):
        def chain(x):
            return torch.relu(x + 1.0)

        tracer = GraphTracer()
        graph = tracer.trace(chain, (torch.randn(4, 4),))
        graph.lint()
        ops = {n.target for n in graph.nodes if n.op == "call_function"}
        # relu and add should be in the graph
        self.assertTrue(len(ops) >= 1)

    def test_trace_matmul(self):
        def matmul(x, y):
            return x @ y

        tracer = GraphTracer()
        graph = tracer.trace(matmul, (torch.randn(4, 8), torch.randn(8, 4)))
        graph.lint()
        call_nodes = [n for n in graph.nodes if n.op == "call_function"]
        self.assertGreater(len(call_nodes), 0)

    def test_trace_multiple_outputs(self):
        def split_heads(x):
            a, b = x.chunk(2, dim=-1)
            return a, b

        tracer = GraphTracer()
        graph = tracer.trace(split_heads, (torch.randn(2, 8),))
        graph.lint()
        # output should be a tuple
        output_node = next(n for n in graph.nodes if n.op == "output")
        self.assertIsInstance(output_node.args[0], (tuple, list))

    def test_trace_nn_linear(self):
        model = torch.nn.Linear(8, 4, bias=True)
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(2, 8),))
        graph.lint()
        ops = [n.op for n in graph.nodes]
        self.assertIn("placeholder", ops)
        self.assertIn("call_function", ops)
        self.assertIn("output", ops)

    def test_trace_nn_relu(self):
        model = torch.nn.ReLU()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(3, 5),))
        graph.lint()

    def test_trace_kwargs(self):
        def add_kw(x, y):
            return x + y

        tracer = GraphTracer()
        graph = tracer.trace(
            add_kw, args=(), kwargs={"x": torch.randn(2, 2), "y": torch.randn(2, 2)}
        )
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        self.assertIn("x", ph_names)
        self.assertIn("y", ph_names)

    def test_trace_dynamic_shapes(self):
        def add(x, y):
            return x + y

        tracer = GraphTracer()
        graph = tracer.trace(
            add,
            (torch.randn(4, 8), torch.randn(4, 8)),
            dynamic_shapes={"x_0": [TracingInt("batch"), 8], "x_1": [TracingInt("batch"), 8]},
        )
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 2)

    def test_trace_model_function(self):
        graph = trace_model(lambda x: x * 2, (torch.randn(3, 3),))
        graph.lint()
        ops = [n.op for n in graph.nodes]
        self.assertIn("call_function", ops)

    def test_trace_model_nn_module(self):
        model = torch.nn.Linear(4, 4)
        graph = trace_model(model, (torch.randn(2, 4),))
        graph.lint()
        self.assertIsNotNone(graph)

    def test_retrace_resets_state(self):
        tracer1 = GraphTracer()
        tracer2 = GraphTracer()

        graph1 = tracer1.trace(lambda x: x + 1, (torch.randn(2, 2),))
        n1 = len(list(graph1.nodes))
        self.assertGreater(n1, 0)

        graph2 = tracer2.trace(lambda x, y: x + y, (torch.randn(2, 2), torch.randn(2, 2)))
        n2 = len(list(graph2.nodes))
        self.assertGreater(n2, 0)

        # second graph should have one more placeholder
        ph1 = [n for n in graph1.nodes if n.op == "placeholder"]
        ph2 = [n for n in graph2.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph1), 1)
        self.assertEqual(len(ph2), 2)
        graph1.lint()
        graph2.lint()

    def test_placeholder_meta_val(self):
        tracer = GraphTracer()
        graph = tracer.trace(lambda x: x, (torch.randn(3, 5),))
        ph = next(n for n in graph.nodes if n.op == "placeholder")
        self.assertIn("val", ph.meta)
        self.assertEqual(ph.meta["val"].shape, torch.Size([3, 5]))

    def test_call_function_meta_val(self):
        tracer = GraphTracer()
        graph = tracer.trace(lambda x: x + 1.0, (torch.randn(3, 5),))
        cf = next(n for n in graph.nodes if n.op == "call_function")
        self.assertIn("val", cf.meta)
        self.assertEqual(cf.meta["val"].shape, torch.Size([3, 5]))

    def test_call_function_meta_stack_trace(self):
        tracer = GraphTracer()
        graph = tracer.trace(lambda x: x + 1.0, (torch.randn(3, 5),))
        cf = next(n for n in graph.nodes if n.op == "call_function")
        self.assertIn("stack_trace", cf.meta)
        self.assertIsInstance(cf.meta["stack_trace"], str)
        # The stack trace should contain file/line information.
        self.assertIn("File", cf.meta["stack_trace"])

    def test_multiple_outputs_getitem_meta_stack_trace(self):
        def split_heads(x):
            a, b = x.chunk(2, dim=-1)
            return a, b

        tracer = GraphTracer()
        graph = tracer.trace(split_heads, (torch.randn(2, 8),))
        graph.lint()
        # getitem nodes should have stack_trace propagated
        getitem_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is operator.getitem
        ]
        self.assertGreater(len(getitem_nodes), 0)
        for n in getitem_nodes:
            self.assertIn("stack_trace", n.meta, f"node {n.name} missing stack_trace")
            self.assertIsInstance(n.meta["stack_trace"], str)

    def test_trace_list_of_tensors(self):
        def add_list(tensors):
            return tensors[0] + tensors[1]

        tracer = GraphTracer()
        graph = tracer.trace(add_list, ([torch.randn(2, 3), torch.randn(2, 3)],))
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        # Two tensors inside the list → two placeholders
        self.assertEqual(len(ph_nodes), 2)
        # Names should encode the nested path
        ph_names = [n.name for n in ph_nodes]
        self.assertIn("tensors_0", ph_names)
        self.assertIn("tensors_1", ph_names)

    def test_trace_tuple_of_tensors(self):
        def add_tuple(pair):
            a, b = pair
            return a + b

        tracer = GraphTracer()
        graph = tracer.trace(add_tuple, ((torch.randn(3, 4), torch.randn(3, 4)),))
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 2)

    def test_trace_dict_of_tensors(self):
        def add_dict(d):
            return d["a"] + d["b"]

        tracer = GraphTracer()
        graph = tracer.trace(add_dict, ({"a": torch.randn(2, 2), "b": torch.randn(2, 2)},))
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 2)
        ph_names = [n.name for n in ph_nodes]
        self.assertIn("d_a", ph_names)
        self.assertIn("d_b", ph_names)

    def test_trace_mixed_nested_input(self):
        def compute(pair, z):
            return pair[0] + pair[1] + z

        tracer = GraphTracer()
        graph = tracer.trace(compute, ([torch.randn(2, 2), torch.randn(2, 2)], torch.randn(2, 2)))
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        # Two from the list + one plain tensor = 3 placeholders
        self.assertEqual(len(ph_nodes), 3)

    def test_trace_nested_scalar_passthrough(self):
        def scale(tensors, factor):
            return tensors[0] * factor

        tracer = GraphTracer()
        graph = tracer.trace(scale, ([torch.randn(3, 3)], 2.0))
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        # Only the tensor inside the list gets a placeholder; 2.0 is a scalar
        self.assertEqual(len(ph_nodes), 1)

    # ------------------------------------------------------------------
    # nn.Module: named parameter / buffer placeholders
    # ------------------------------------------------------------------

    def test_trace_nn_module_named_weight_placeholders(self):
        model = torch.nn.Linear(4, 2, bias=True)
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(3, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # Expect "weight" and "bias" placeholder nodes (sanitized module param names)
        self.assertTrue(
            any("weight" in n for n in ph_names), f"No 'weight' placeholder found in {ph_names}"
        )
        self.assertTrue(
            any("bias" in n for n in ph_names), f"No 'bias' placeholder found in {ph_names}"
        )

    def test_trace_nn_module_shared_parameter_single_placeholder(self):
        class SharedWeight(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                # Use self.w in two matmuls — same tensor object both times
                return x @ self.w + x @ self.w

        model = SharedWeight()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # Only one placeholder for 'w', not two
        w_phs = [n for n in ph_names if "w" in n]
        self.assertEqual(len(w_phs), 1, f"Expected 1 'w' placeholder, got {w_phs}")

    def test_trace_nested_module_parameter_names(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # nn.Sequential(Linear(4,4)) has params "0.weight" and "0.bias",
        # sanitized to "0_weight" and "0_bias"
        self.assertTrue(
            any("weight" in n for n in ph_names), f"No 'weight' placeholder found in {ph_names}"
        )

    def test_trace_custom_nn_module(self):
        class TwoLayerMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(8, 4, bias=True)
                self.fc2 = torch.nn.Linear(4, 2, bias=True)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = TwoLayerMLP()
        tracer = GraphTracer(verbose=3)
        graph = tracer.trace(model, (torch.randn(3, 8),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # 4 parameters: fc1.weight, fc1.bias, fc2.weight, fc2.bias + 1 input
        self.assertGreaterEqual(len(ph_names), 5)
        self.assertTrue(any("weight" in n for n in ph_names))
        self.assertTrue(any("bias" in n for n in ph_names))
        call_nodes = [n for n in graph.nodes if n.op == "call_function"]
        self.assertGreater(len(call_nodes), 0)

    def test_trace_model_with_nn_module(self):
        model = torch.nn.Linear(4, 2, bias=True)
        graph = trace_model(model, (torch.randn(1, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        self.assertTrue(any("weight" in n for n in ph_names))
        self.assertTrue(any("bias" in n for n in ph_names))

    def test_trace_nn_module_with_buffer(self):
        model = torch.nn.BatchNorm1d(4)
        model.eval()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(3, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # BatchNorm has weight/bias params and running_mean/running_var buffers
        self.assertTrue(
            any("running_mean" in n or "running_var" in n for n in ph_names),
            f"No buffer placeholders found in {ph_names}",
        )

    def test_trace_nn_module_parameter_count(self):
        model = torch.nn.Linear(4, 2, bias=True)
        param_names = [name for name, _ in model.named_parameters()]
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # Each named parameter should have exactly one placeholder node
        for param_name in param_names:
            sanitized = param_name.replace(".", "_")
            self.assertIn(
                sanitized, ph_names, f"Expected placeholder '{sanitized}' in {ph_names}"
            )

    def test_trace_nn_module_no_bias(self):
        model = torch.nn.Linear(4, 2, bias=False)
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        self.assertTrue(any("weight" in n for n in ph_names))
        # No bias placeholder expected
        self.assertFalse(
            any("bias" in n for n in ph_names), f"Unexpected 'bias' placeholder in {ph_names}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
