"""Tests for yobx.torch.new_tracing."""

import operator
import unittest
import torch
from yobx.ext_test_case import ExtTestCase, requires_torch


@requires_torch("2.0")
class TestNewTracing(ExtTestCase):
    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def test_import(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer
        from yobx.torch.new_tracing.shape import TracingBool, TracingInt, TracingShape
        from yobx.torch.new_tracing.tensor import TracingTensor
        from yobx.torch.new_tracing import trace_model

        self.assertIsNotNone(DispatchTracer)
        self.assertIsNotNone(TracingBool)
        self.assertIsNotNone(TracingInt)
        self.assertIsNotNone(TracingShape)
        self.assertIsNotNone(TracingTensor)
        self.assertIsNotNone(trace_model)

    # ------------------------------------------------------------------
    # TracingInt / TracingBool
    # ------------------------------------------------------------------

    def test_tracing_dimension_repr_no_value(self):
        from yobx.torch.new_tracing.shape import TracingInt

        d = TracingInt("batch")
        self.assertIn("batch", repr(d))
        self.assertEqual(str(d), "batch")

    def test_tracing_dimension_repr_with_value(self):
        from yobx.torch.new_tracing.shape import TracingInt

        d = TracingInt(4)
        self.assertIn("4", repr(d))
        self.assertEqual(int(d), 4)

    def test_tracing_dimension_int_raises_without_value(self):
        from yobx.torch.new_tracing.shape import TracingInt

        d = TracingInt("n")
        with self.assertRaises(ValueError):
            int(d)

    def test_tracing_dimension_eq(self):
        from yobx.torch.new_tracing.shape import TracingBool, TracingInt

        # Concrete comparison → plain bool
        d1 = TracingInt(4)
        d2 = TracingInt(4)
        d3 = TracingInt(8)
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, d3)
        self.assertEqual(d1, 4)  # compare concrete TracingInt with plain int

        # Symbolic comparison → TracingBool
        d_sym = TracingInt("batch")
        result = d_sym == 4
        self.assertIsInstance(result, TracingBool)
        self.assertIn("batch", str(result.value))

    def test_tracing_dimension_arithmetic(self):
        from yobx.torch.new_tracing.shape import TracingInt

        # Concrete arithmetic produces concrete TracingInt
        d = TracingInt(8)
        self.assertEqual(int(d + 2), 10)
        self.assertEqual(int(d - 3), 5)
        self.assertEqual(int(d * 2), 16)
        self.assertEqual(int(d // 4), 2)
        self.assertEqual(int(2 + d), 10)
        self.assertEqual(int(2 * d), 16)

        # Symbolic arithmetic preserves the expression as a string value
        s = TracingInt("n")
        result = s + 2
        self.assertIsInstance(result, TracingInt)
        self.assertIn("n", str(result.value))

    def test_tracing_dimension_neg(self):
        from yobx.torch.new_tracing.shape import TracingInt

        d = TracingInt(5)
        negd = -d
        self.assertEqual(int(negd), -5)

    def test_tracing_dimension_hash(self):
        from yobx.torch.new_tracing.shape import TracingInt

        d = TracingInt("batch")
        self.assertIsInstance(hash(d), int)
        self.assertIn(d, {d})

    def test_tracing_bool_concrete(self):
        from yobx.torch.new_tracing.shape import TracingBool

        tb_true = TracingBool(True)
        tb_false = TracingBool(False)
        self.assertTrue(bool(tb_true))
        self.assertFalse(bool(tb_false))

    def test_tracing_bool_symbolic_raises_on_bool(self):
        from yobx.torch.new_tracing.shape import TracingBool

        tb = TracingBool("(n==4)")
        with self.assertRaises(ValueError):
            bool(tb)

    # ------------------------------------------------------------------
    # TracingShape
    # ------------------------------------------------------------------

    def test_tracing_shape_concrete(self):
        from yobx.torch.new_tracing.shape import TracingInt, TracingShape

        s = TracingShape([TracingInt(4), 16])
        self.assertTrue(s.is_concrete)
        self.assertEqual(s.numel(), 64)
        self.assertEqual(s.to_torch_size(), torch.Size([4, 16]))

    def test_tracing_shape_symbolic(self):
        from yobx.torch.new_tracing.shape import TracingInt, TracingShape

        s = TracingShape([TracingInt("n"), 8])
        self.assertFalse(s.is_concrete)
        with self.assertRaises(ValueError):
            s.numel()
        with self.assertRaises(ValueError):
            s.to_torch_size()

    def test_tracing_shape_repr(self):
        from yobx.torch.new_tracing.shape import TracingInt, TracingShape

        s = TracingShape([TracingInt(2), 4])
        r = repr(s)
        self.assertIn("TracingShape", r)

    def test_tracing_shape_indexing(self):
        from yobx.torch.new_tracing.shape import TracingInt, TracingShape

        d = TracingInt(5)
        s = TracingShape([d, 8])
        self.assertIs(s[0], d)
        self.assertEqual(s[1], 8)

    # ------------------------------------------------------------------
    # TracingTensor
    # ------------------------------------------------------------------

    def test_tracing_tensor_creation(self):
        from yobx.torch.new_tracing.tensor import TracingTensor

        t = TracingTensor.__new__(TracingTensor, (3, 4), dtype=torch.float32)
        t.__init__((3, 4), dtype=torch.float32)
        self.assertEqual(t.shape, torch.Size([3, 4]))
        self.assertEqual(t.dtype, torch.float32)

    def test_tracing_tensor_repr(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        tracer = DispatchTracer()
        x = torch.randn(2, 3)
        t = tracer.placeholder("x", x.shape, x.dtype, x.device)
        self.assertIn("TracingTensor", repr(t))
        self.assertIn("x", repr(t))

    # ------------------------------------------------------------------
    # DispatchTracer – basic graphs
    # ------------------------------------------------------------------

    def test_trace_simple_add(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def add(x, y):
            return x + y

        tracer = DispatchTracer()
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
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def mul(x, y):
            return x * y

        tracer = DispatchTracer()
        graph = tracer.trace(mul, (torch.randn(2, 5), torch.randn(2, 5)))
        graph.lint()
        call_nodes = [n for n in graph.nodes if n.op == "call_function"]
        self.assertGreater(len(call_nodes), 0)

    def test_trace_elementwise_chain(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def chain(x):
            return torch.relu(x + 1.0)

        tracer = DispatchTracer()
        graph = tracer.trace(chain, (torch.randn(4, 4),))
        graph.lint()
        ops = {n.target for n in graph.nodes if n.op == "call_function"}
        # relu and add should be in the graph
        self.assertTrue(len(ops) >= 1)

    def test_trace_matmul(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def matmul(x, y):
            return x @ y

        tracer = DispatchTracer()
        graph = tracer.trace(matmul, (torch.randn(4, 8), torch.randn(8, 4)))
        graph.lint()
        call_nodes = [n for n in graph.nodes if n.op == "call_function"]
        self.assertGreater(len(call_nodes), 0)

    def test_trace_multiple_outputs(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def split_heads(x):
            a, b = x.chunk(2, dim=-1)
            return a, b

        tracer = DispatchTracer()
        graph = tracer.trace(split_heads, (torch.randn(2, 8),))
        graph.lint()
        # output should be a tuple
        output_node = next(n for n in graph.nodes if n.op == "output")
        self.assertIsInstance(output_node.args[0], (tuple, list))

    def test_trace_nn_linear(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        model = torch.nn.Linear(8, 4, bias=True)
        tracer = DispatchTracer()
        graph = tracer.trace(model, (torch.randn(2, 8),))
        graph.lint()
        ops = [n.op for n in graph.nodes]
        self.assertIn("placeholder", ops)
        self.assertIn("call_function", ops)
        self.assertIn("output", ops)

    def test_trace_nn_relu(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        model = torch.nn.ReLU()
        tracer = DispatchTracer()
        graph = tracer.trace(model, (torch.randn(3, 5),))
        graph.lint()

    def test_trace_kwargs(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def add_kw(x, y):
            return x + y

        tracer = DispatchTracer()
        graph = tracer.trace(
            add_kw, args=(), kwargs={"x": torch.randn(2, 2), "y": torch.randn(2, 2)}
        )
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        self.assertIn("x", ph_names)
        self.assertIn("y", ph_names)

    # ------------------------------------------------------------------
    # DispatchTracer – dynamic shapes
    # ------------------------------------------------------------------

    def test_trace_dynamic_shapes(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer, TracingInt

        def add(x, y):
            return x + y

        tracer = DispatchTracer()
        graph = tracer.trace(
            add,
            (torch.randn(4, 8), torch.randn(4, 8)),
            dynamic_shapes={"x_0": [TracingInt("batch"), 8], "x_1": [TracingInt("batch"), 8]},
        )
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 2)

    # ------------------------------------------------------------------
    # trace_model convenience function
    # ------------------------------------------------------------------

    def test_trace_model_function(self):
        from yobx.torch.new_tracing import trace_model

        graph = trace_model(lambda x: x * 2, (torch.randn(3, 3),))
        graph.lint()
        ops = [n.op for n in graph.nodes]
        self.assertIn("call_function", ops)

    def test_trace_model_nn_module(self):
        from yobx.torch.new_tracing import trace_model

        model = torch.nn.Linear(4, 4)
        graph = trace_model(model, (torch.randn(2, 4),))
        graph.lint()
        self.assertIsNotNone(graph)

    # ------------------------------------------------------------------
    # Re-tracing: calling trace() twice on the same DispatchTracer
    # ------------------------------------------------------------------

    def test_retrace_resets_state(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        tracer = DispatchTracer()

        graph1 = tracer.trace(lambda x: x + 1, (torch.randn(2, 2),))
        n1 = len(list(graph1.nodes))
        self.assertGreater(n1, 0)

        graph2 = tracer.trace(lambda x, y: x + y, (torch.randn(2, 2), torch.randn(2, 2)))
        n2 = len(list(graph2.nodes))
        self.assertGreater(n2, 0)

        # second graph should have one more placeholder
        ph1 = [n for n in graph1.nodes if n.op == "placeholder"]
        ph2 = [n for n in graph2.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph1), 1)
        self.assertEqual(len(ph2), 2)
        graph1.lint()
        graph2.lint()

    # ------------------------------------------------------------------
    # Graph node metadata
    # ------------------------------------------------------------------

    def test_placeholder_meta_val(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        tracer = DispatchTracer()
        graph = tracer.trace(lambda x: x, (torch.randn(3, 5),))
        ph = next(n for n in graph.nodes if n.op == "placeholder")
        self.assertIn("val", ph.meta)
        self.assertEqual(ph.meta["val"].shape, torch.Size([3, 5]))

    def test_call_function_meta_val(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        tracer = DispatchTracer()
        graph = tracer.trace(lambda x: x + 1.0, (torch.randn(3, 5),))
        cf = next(n for n in graph.nodes if n.op == "call_function")
        self.assertIn("val", cf.meta)
        self.assertEqual(cf.meta["val"].shape, torch.Size([3, 5]))

    def test_call_function_meta_stack_trace(self):
        """call_function nodes should have a 'stack_trace' entry in node.meta."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        tracer = DispatchTracer()
        graph = tracer.trace(lambda x: x + 1.0, (torch.randn(3, 5),))
        cf = next(n for n in graph.nodes if n.op == "call_function")
        self.assertIn("stack_trace", cf.meta)
        self.assertIsInstance(cf.meta["stack_trace"], str)
        # The stack trace should contain file/line information.
        self.assertIn("File", cf.meta["stack_trace"])

    def test_multiple_outputs_getitem_meta_stack_trace(self):
        """getitem nodes for multi-output ops inherit 'stack_trace' from their parent."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def split_heads(x):
            a, b = x.chunk(2, dim=-1)
            return a, b

        tracer = DispatchTracer()
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

    # ------------------------------------------------------------------
    # Nested input structures
    # ------------------------------------------------------------------

    def test_trace_list_of_tensors(self):
        """Inputs as a list of tensors should produce one placeholder per tensor."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def add_list(tensors):
            return tensors[0] + tensors[1]

        tracer = DispatchTracer()
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
        """Inputs as a tuple of tensors should produce one placeholder per tensor."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def add_tuple(pair):
            a, b = pair
            return a + b

        tracer = DispatchTracer()
        graph = tracer.trace(add_tuple, ((torch.randn(3, 4), torch.randn(3, 4)),))
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 2)

    def test_trace_dict_of_tensors(self):
        """Inputs as a dict of tensors should produce one placeholder per tensor."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def add_dict(d):
            return d["a"] + d["b"]

        tracer = DispatchTracer()
        graph = tracer.trace(add_dict, ({"a": torch.randn(2, 2), "b": torch.randn(2, 2)},))
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 2)
        ph_names = [n.name for n in ph_nodes]
        self.assertIn("d_a", ph_names)
        self.assertIn("d_b", ph_names)

    def test_trace_mixed_nested_input(self):
        """Mixed nesting: top-level tuple arg containing a list and a plain tensor."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def compute(pair, z):
            return pair[0] + pair[1] + z

        tracer = DispatchTracer()
        graph = tracer.trace(compute, ([torch.randn(2, 2), torch.randn(2, 2)], torch.randn(2, 2)))
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        # Two from the list + one plain tensor = 3 placeholders
        self.assertEqual(len(ph_nodes), 3)

    def test_trace_nested_scalar_passthrough(self):
        """Non-tensor values inside a nested structure pass through unchanged."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        def scale(tensors, factor):
            return tensors[0] * factor

        tracer = DispatchTracer()
        graph = tracer.trace(scale, ([torch.randn(3, 3)], 2.0))
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        # Only the tensor inside the list gets a placeholder; 2.0 is a scalar
        self.assertEqual(len(ph_nodes), 1)

    # ------------------------------------------------------------------
    # nn.Module: named parameter / buffer placeholders
    # ------------------------------------------------------------------

    def test_trace_nn_module_named_weight_placeholders(self):
        """Parameters of an nn.Module get named placeholder nodes."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        model = torch.nn.Linear(4, 2, bias=True)
        tracer = DispatchTracer()
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
        """A parameter used twice maps to exactly one placeholder (no duplicates)."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        class SharedWeight(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                # Use self.w in two matmuls — same tensor object both times
                return x @ self.w + x @ self.w

        model = SharedWeight()
        tracer = DispatchTracer()
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # Only one placeholder for 'w', not two
        w_phs = [n for n in ph_names if "w" in n]
        self.assertEqual(len(w_phs), 1, f"Expected 1 'w' placeholder, got {w_phs}")

    def test_trace_nested_module_parameter_names(self):
        """Nested module parameters appear with sanitized dotted names."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        tracer = DispatchTracer()
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # nn.Sequential(Linear(4,4)) has params "0.weight" and "0.bias",
        # sanitized to "0_weight" and "0_bias"
        self.assertTrue(
            any("weight" in n for n in ph_names), f"No 'weight' placeholder found in {ph_names}"
        )

    # ------------------------------------------------------------------
    # nn.Module: custom subclasses and functional tests
    # ------------------------------------------------------------------

    def test_trace_custom_nn_module(self):
        """Tracing a custom nn.Module subclass produces a valid graph."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        class TwoLayerMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(8, 4, bias=True)
                self.fc2 = torch.nn.Linear(4, 2, bias=True)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = TwoLayerMLP()
        tracer = DispatchTracer()
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
        """trace_model convenience wrapper works with an nn.Module."""
        from yobx.torch.new_tracing import trace_model

        model = torch.nn.Linear(4, 2, bias=True)
        graph = trace_model(model, (torch.randn(1, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        self.assertTrue(any("weight" in n for n in ph_names))
        self.assertTrue(any("bias" in n for n in ph_names))

    def test_trace_nn_module_with_buffer(self):
        """Buffers of an nn.Module get named placeholder nodes."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        model = torch.nn.BatchNorm1d(4)
        model.eval()
        tracer = DispatchTracer()
        graph = tracer.trace(model, (torch.randn(3, 4),))
        graph.lint()
        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # BatchNorm has weight/bias params and running_mean/running_var buffers
        self.assertTrue(
            any("running_mean" in n or "running_var" in n for n in ph_names),
            f"No buffer placeholders found in {ph_names}",
        )

    def test_trace_nn_module_parameter_count(self):
        """The number of parameter placeholders matches named_parameters() count."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        model = torch.nn.Linear(4, 2, bias=True)
        param_names = [name for name, _ in model.named_parameters()]
        tracer = DispatchTracer()
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
        """nn.Module without bias produces a graph without a bias placeholder."""
        from yobx.torch.new_tracing.dispatcher import DispatchTracer

        model = torch.nn.Linear(4, 2, bias=False)
        tracer = DispatchTracer()
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
