"""Tests for yobx.torch.new_tracing – tracing (GraphTracer and trace_model)."""

import operator
import unittest
import torch
from yobx.ext_test_case import ExtTestCase, hide_stdout
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

    def test_trace_inplace_iadd(self):
        """Inplace += on a module parameter must appear in the output, not be dead code."""

        class InplaceAddModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.ones((1, 4), dtype=torch.float32)

            def forward(self, x):
                x += self.bias
                return x

        model = InplaceAddModel()
        x = torch.rand(3, 4)

        tracer = GraphTracer()
        graph = tracer.trace(model, (x.clone(),))
        graph.lint()

        # The inplace add_ should be present and the output should reference it.
        call_nodes = [n for n in graph.nodes if n.op == "call_function"]
        self.assertGreater(len(call_nodes), 0, "Expected at least one call_function node")
        output_node = next(n for n in graph.nodes if n.op == "output")
        # The output must not be the bare input placeholder — it must use the add result.
        result_node = output_node.args[0]
        self.assertIsNotNone(result_node)
        self.assertNotEqual(result_node.op, "placeholder", "Output must not be the raw input")

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
            dynamic_shapes={"x": {0: "batch"}, "y": {0: "batch"}},
        )
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 2)

    def test_trace_dynamic_shapes_torch_export_dim(self):
        """GraphTracer accepts torch.export.Dim objects in dynamic_shapes."""

        def add(x, y):
            return x + y

        batch = torch.export.Dim("batch")
        tracer = GraphTracer()
        graph = tracer.trace(
            add,
            (torch.randn(4, 8), torch.randn(4, 8)),
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 2)
        # The placeholder meta shapes should be symbolic (TracingInt).
        from yobx.torch.new_tracing.shape import TracingInt

        for node in ph_nodes:
            shape = node.meta["val"].shape
            self.assertIsInstance(shape.dims[0], TracingInt)
            self.assertEqual(str(shape.dims[0]), "batch")

    def test_trace_dynamic_shapes_dim_dynamic(self):
        """GraphTracer accepts unnamed Dim hints (Dim.DYNAMIC) in dynamic_shapes."""

        def add(x, y):
            return x + y

        tracer = GraphTracer()
        graph = tracer.trace(
            add,
            (torch.randn(4, 8), torch.randn(4, 8)),
            dynamic_shapes={
                "x": {0: torch.export.Dim.DYNAMIC},
                "y": {0: torch.export.Dim.DYNAMIC},
            },
        )
        graph.lint()
        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 2)
        from yobx.torch.new_tracing.shape import TracingInt

        for node in ph_nodes:
            shape = node.meta["val"].shape
            self.assertIsInstance(shape.dims[0], TracingInt)

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

    def test_call_function_meta_fn(self):
        tracer = GraphTracer()
        graph = tracer.trace(lambda x: x + 1.0, (torch.randn(3, 5),))
        cf = next(n for n in graph.nodes if n.op == "call_function")
        self.assertIn("fn", cf.meta)
        # No nn.Module on the stack for a plain lambda: meta["fn"] must be None.
        self.assertIsNone(cf.meta["fn"])

    def test_call_function_meta_fn_module(self):
        model = torch.nn.Linear(4, 2, bias=False)
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(2, 4),))
        cf = next(n for n in graph.nodes if n.op == "call_function")
        self.assertIn("fn", cf.meta)
        # The stored module must be the nn.Module responsible for the op.
        self.assertIsInstance(cf.meta["fn"], torch.nn.Module)

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

    @hide_stdout()
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

    # ------------------------------------------------------------------
    # torch.cond support
    # ------------------------------------------------------------------

    def test_trace_cond_single_output(self):
        """torch.cond with a single tensor output is captured as a call_function node."""

        class CondModel(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x)

                def false_fn(x):
                    return torch.cos(x)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        model = CondModel()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(5, 3),))
        graph.lint()

        # Exactly one call_function node with target torch.cond
        cond_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is torch.cond
        ]
        self.assertEqual(len(cond_nodes), 1, f"Expected 1 cond node, got: {cond_nodes}")

        # Branch sub-graphs are stored in _sub_tracers
        self.assertIn("_cb_cond_true_fn_0", tracer._sub_tracers)
        self.assertIn("_cb_cond_false_fn_0", tracer._sub_tracers)

        # Sub-graphs should be valid FX graphs
        for _name, sub in tracer._sub_tracers.items():
            sub.graph.lint()

    def test_trace_cond_two_outputs(self):
        """torch.cond whose branches return a tuple produces getitem nodes."""

        class TwoOutModel(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x), torch.cos(x)

                def false_fn(x):
                    return torch.cos(x), torch.sin(x)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        model = TwoOutModel()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(5, 3),))
        graph.lint()

        # One cond node + two getitem nodes
        cond_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is torch.cond
        ]
        self.assertEqual(len(cond_nodes), 1)
        gi_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is operator.getitem
        ]
        self.assertGreaterEqual(len(gi_nodes), 2)

    def test_trace_cond_two_inputs(self):
        """torch.cond with two tensor operands produces a cond node."""

        class TwoInputModel(torch.nn.Module):
            def forward(self, x, y):
                def true_fn(x, y):
                    return torch.sin(x), torch.cos(x) + y

                def false_fn(x, y):
                    return torch.cos(x), torch.sin(x) + y

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x, y])

        x, y = torch.randn(5, 3), torch.randn(5, 3)
        model = TwoInputModel()
        tracer = GraphTracer()
        graph = tracer.trace(model, (x, y))
        graph.lint()

        cond_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is torch.cond
        ]
        self.assertEqual(len(cond_nodes), 1)

    def test_trace_cond_nested(self):
        """Nested torch.cond calls are each captured as separate cond nodes."""

        class NestedCondModel(torch.nn.Module):
            def forward(self, x):
                def true_fn2(x):
                    def true_fn1(x):
                        return torch.sin(x)

                    def false_fn1(x):
                        return torch.cos(x)

                    return torch.cond(x.sum() < 0, true_fn1, false_fn1, [x])

                def false_fn2(x):
                    return -x

                return torch.cond(x.sum() > 0, true_fn2, false_fn2, [x])

        model = NestedCondModel()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(5, 3),))
        graph.lint()

        # Main graph has one cond node
        cond_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is torch.cond
        ]
        self.assertEqual(len(cond_nodes), 1)

        # The true_fn2 sub-graph itself also contains a cond node
        true_sub = tracer._sub_tracers.get("_cb_cond_true_fn2_0")
        self.assertIsNotNone(true_sub, f"Keys: {list(tracer._sub_tracers)}")
        nested_cond = [
            n for n in true_sub.graph.nodes if n.op == "call_function" and n.target is torch.cond
        ]
        self.assertEqual(len(nested_cond), 1)

    def test_trace_cond_with_module_params(self):
        """torch.cond branches that reference nn.Module parameters are traced correctly."""

        class WeightedCond(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([42.0]))

            def forward(self, x):
                def true_fn(x):
                    return x + self.weight

                def false_fn(x):
                    return x - self.weight

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        model = WeightedCond()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(3, 4),))
        graph.lint()

        cond_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is torch.cond
        ]
        self.assertEqual(len(cond_nodes), 1)

        # Branch sub-graphs reference a parameter placeholder
        true_sub = tracer._sub_tracers["_cb_cond_true_fn_0"]
        true_sub.graph.lint()
        # There should be a placeholder for the weight parameter in the sub-graph
        ph_names = [n.name for n in true_sub.graph.nodes if n.op == "placeholder"]
        self.assertTrue(
            any("param" in n or "operand" in n for n in ph_names),
            f"No parameter placeholder in sub-graph: {ph_names}",
        )

    def test_trace_cond_with_constant_ones(self):
        """torch.cond branch using torch.ones with a constant shape is traced correctly.

        Regression test for ControlFlowCondConstant: the false_fn calls
        ``torch.ones((1, N), ...)`` with a fully concrete integer shape.  Previously
        _handle_ones early-exited and returned a plain tensor, which then appeared as
        an ``param_N`` placeholder in the branch sub-graph.  With the fix, _handle_ones
        always emits an FX node so the branch sub-graph stays self-contained.
        """

        class ConstantOnesModel(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x) - torch.ones(x.shape, dtype=x.dtype, device=x.device)

                def false_fn(x):
                    return torch.cos(x) + torch.ones((1, 4), dtype=x.dtype, device=x.device)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        model = ConstantOnesModel()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()

        cond_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is torch.cond
        ]
        self.assertEqual(len(cond_nodes), 1)

        # Both branch sub-graphs should be valid
        self.assertIn("_cb_cond_true_fn_0", tracer._sub_tracers)
        self.assertIn("_cb_cond_false_fn_0", tracer._sub_tracers)
        for _name, sub in tracer._sub_tracers.items():
            sub.graph.lint()

        # The false branch should contain a call_function node for torch.ones,
        # not an extra placeholder for the constant tensor.
        from yobx.torch.new_tracing._patches import _ORIGINAL_TORCH_ONES

        false_sub = tracer._sub_tracers["_cb_cond_false_fn_0"]
        ones_nodes = [
            n
            for n in false_sub.graph.nodes
            if n.op == "call_function" and n.target is _ORIGINAL_TORCH_ONES
        ]
        self.assertGreaterEqual(
            len(ones_nodes), 1, "Expected at least one ones() FX node in false branch sub-graph"
        )

    # ------------------------------------------------------------------
    # torch.ops.higher_order.scan support
    # ------------------------------------------------------------------

    @unittest.skipIf(
        not hasattr(getattr(torch.ops, "higher_order", None), "scan"),
        "torch.ops.higher_order.scan not available",
    )
    def test_trace_scan_single_carry_single_output(self):
        """torch.ops.higher_order.scan is captured as a call_function node."""

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                def add(carry, y):
                    next_carry = carry + y
                    return [next_carry, next_carry]

                init = torch.zeros_like(x[0])
                carry, _out = torch.ops.higher_order.scan(add, [init], [x], additional_inputs=[])
                return carry

        model = ScanModel()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(3, 4),))
        graph.lint()

        # Exactly one call_function node with scan target
        scan_op = torch.ops.higher_order.scan
        scan_nodes = [n for n in graph.nodes if n.op == "call_function" and n.target is scan_op]
        self.assertEqual(len(scan_nodes), 1, f"Expected 1 scan node, got: {scan_nodes}")

        # The scan body should be registered
        self.assertTrue(
            any("scan" in k for k in tracer._sub_tracers),
            f"No scan sub-tracer found in {list(tracer._sub_tracers)}",
        )

    @unittest.skipIf(
        not hasattr(getattr(torch.ops, "higher_order", None), "scan"),
        "torch.ops.higher_order.scan not available",
    )
    def test_trace_scan_two_carries_two_scan_inputs(self):
        """GraphTracer correctly traces a scan with 2 carry states and 2 scan inputs."""

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                def add(carry1, carry2, y1, y2):
                    next_carry1 = carry1 + y1
                    next_carry2 = carry2 * y2
                    return [next_carry1, next_carry2, next_carry1, next_carry2]

                init1 = x.new_zeros(x.shape[1:])
                init2 = x.new_zeros(x.shape[1:]) + 1
                carry1, carry2, out1, out2 = torch.ops.higher_order.scan(
                    add, [init1, init2], [x, x * 2], additional_inputs=[]
                )
                return carry1, carry2, out1, out2

        model = ScanModel()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(3, 4),))
        graph.lint()

        scan_op = torch.ops.higher_order.scan
        scan_nodes = [n for n in graph.nodes if n.op == "call_function" and n.target is scan_op]
        self.assertEqual(len(scan_nodes), 1, f"Expected 1 scan node, got: {scan_nodes}")

        # Exactly 4 getitem nodes (2 carry + 2 scan-accumulation outputs).
        getitem_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target is operator.getitem
            and n.args[0] is scan_nodes[0]
        ]
        self.assertEqual(len(getitem_nodes), 4, f"Expected 4 getitem nodes, got: {getitem_nodes}")

        # meta["val"] carries shape info for all 4 outputs.
        val = scan_nodes[0].meta.get("val", None)
        self.assertIsNotNone(val, "scan node must have meta['val'] set")
        self.assertEqual(len(val), 4, f"Expected 4 outputs in meta['val'], got {len(val)}")

        # The scan body sub-tracer is registered.
        self.assertTrue(
            any("scan" in k for k in tracer._sub_tracers),
            f"No scan sub-tracer found in {list(tracer._sub_tracers)}",
        )

    @unittest.skipIf(
        not hasattr(getattr(torch.ops, "higher_order", None), "scan"),
        "torch.ops.higher_order.scan not available",
    )
    def test_trace_scan_cdist(self):
        """ControlFlowScanCDist-like model is correctly traced."""

        class ScanCDistModel(torch.nn.Module):
            def forward(self, x):
                def dist(carry, xi):
                    sub = carry - xi.reshape((1, -1))
                    sq = sub * sub
                    rd = sq.sum(dim=1) ** 0.5
                    return [carry.clone(), rd]

                _carry, out = torch.ops.higher_order.scan(dist, [x], [x], additional_inputs=[])
                return out

        model = ScanCDistModel()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(3, 4),))
        graph.lint()

        scan_op = torch.ops.higher_order.scan
        scan_nodes = [n for n in graph.nodes if n.op == "call_function" and n.target is scan_op]
        self.assertEqual(len(scan_nodes), 1)

        # The scan body sub-tracer is registered
        self.assertTrue(
            any("scan" in k for k in tracer._sub_tracers),
            f"No scan sub-tracer: {list(tracer._sub_tracers)}",
        )

        # get_attr node for the callable exists
        get_attr_nodes = [n for n in graph.nodes if n.op == "get_attr"]
        self.assertGreater(len(get_attr_nodes), 0, "No get_attr node for scan callable")

    @unittest.skipIf(
        not hasattr(getattr(torch.ops, "higher_order", None), "scan"),
        "torch.ops.higher_order.scan not available",
    )
    def test_trace_patched_vmap(self):
        """patched_vmap routes TracingTensor dynamic batch dims to scan."""
        from yobx.torch.testing._model_eval_cases import patched_vmap

        def _mul_plus_one(a, b):
            return a * b + 1

        class VmapModel(torch.nn.Module):
            def forward(self, x, y):
                return patched_vmap(_mul_plus_one)(x, y)

        DYN = torch.export.Dim.DYNAMIC
        model = VmapModel()
        tracer = GraphTracer()
        graph = tracer.trace(
            model,
            (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.1, 0.2, 0.3])),
            dynamic_shapes={"x": {0: DYN}, "y": {0: DYN}},
        )
        graph.lint()

        # Exactly one scan call_function node.
        scan_op = torch.ops.higher_order.scan
        scan_nodes = [n for n in graph.nodes if n.op == "call_function" and n.target is scan_op]
        self.assertEqual(len(scan_nodes), 1, f"Expected 1 scan node, got: {scan_nodes}")

        # The scan body sub-tracer is registered.
        self.assertTrue(
            any("scan" in k for k in tracer._sub_tracers),
            f"No scan sub-tracer found in {list(tracer._sub_tracers)}",
        )

    # ------------------------------------------------------------------
    # module_leaves support
    # ------------------------------------------------------------------

    def test_trace_module_leaves_no_params(self):
        """A leaf module with no parameters emits a single call_function node."""

        class LeafModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = LeafModule()

            def forward(self, x):
                return self.leaf(x) + 1

        model = OuterModule()
        tracer = GraphTracer(
            module_leaves={LeafModule: lambda m, module_qualified_name=None: True}
        )
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()

        # The leaf module should appear as a single call_function node.
        leaf_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is model.leaf
        ]
        self.assertEqual(len(leaf_nodes), 1, f"Expected 1 leaf node, got: {leaf_nodes}")

    def test_trace_module_leaves_with_params_not_in_graph(self):
        """Parameters of a leaf module must NOT appear as graph placeholders."""

        class LeafModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(x)

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = LeafModule()

            def forward(self, x):
                return self.leaf(x) + 1

        model = OuterModule()
        tracer = GraphTracer(
            module_leaves={LeafModule: lambda m, module_qualified_name=None: True}
        )
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()

        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # The input tensor should be a placeholder; no leaf-module params.
        self.assertFalse(
            any("fc" in n for n in ph_names),
            f"Leaf module parameters should not be graph placeholders, got: {ph_names}",
        )

        # The leaf module itself must appear as a call_function node.
        leaf_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is model.leaf
        ]
        self.assertEqual(len(leaf_nodes), 1)

    def test_trace_module_leaves_outer_params_preserved(self):
        """Parameters of the *outer* module must still be registered."""

        class LeafModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(x)

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = LeafModule()
                self.bias = torch.nn.Parameter(torch.zeros(4))

            def forward(self, x):
                return self.leaf(x) + self.bias

        model = OuterModule()
        tracer = GraphTracer(
            module_leaves={LeafModule: lambda m, module_qualified_name=None: True}
        )
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()

        ph_names = [n.name for n in graph.nodes if n.op == "placeholder"]
        # The outer module's own parameter should be present.
        self.assertIn("bias", ph_names, f"Outer module bias not found in {ph_names}")
        # Leaf module parameters should be absent.
        self.assertFalse(
            any("fc" in n for n in ph_names),
            f"Leaf module parameters should not appear: {ph_names}",
        )

    def test_trace_module_leaves_predicate_false(self):
        """When the predicate returns False the module is traced normally."""

        class ConditionalLeaf(torch.nn.Module):
            def forward(self, x):
                return x * 3

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = ConditionalLeaf()

            def forward(self, x):
                return self.sub(x) + 1

        def never_leaf(m, module_qualified_name=None):
            return False

        model = OuterModule()
        tracer = GraphTracer(module_leaves={ConditionalLeaf: never_leaf})
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()

        # With predicate returning False the module is traced through; no
        # call_function node with model.sub as target should exist.
        leaf_nodes = [n for n in graph.nodes if n.op == "call_function" and n.target is model.sub]
        self.assertEqual(len(leaf_nodes), 0, f"Expected no leaf node, got: {leaf_nodes}")

    def test_trace_module_leaves_multiple(self):
        """Multiple leaf sub-modules each become a single call_function node."""

        class LeafA(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class LeafB(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = LeafA()
                self.b = LeafB()

            def forward(self, x):
                return self.a(x) + self.b(x)

        model = OuterModule()
        tracer = GraphTracer(
            module_leaves={
                LeafA: lambda m, module_qualified_name=None: True,
                LeafB: lambda m, module_qualified_name=None: True,
            }
        )
        graph = tracer.trace(model, (torch.randn(2, 4),))
        graph.lint()

        leaf_a_nodes = [n for n in graph.nodes if n.op == "call_function" and n.target is model.a]
        leaf_b_nodes = [n for n in graph.nodes if n.op == "call_function" and n.target is model.b]
        self.assertEqual(len(leaf_a_nodes), 1)
        self.assertEqual(len(leaf_b_nodes), 1)

    def test_trace_model_with_module_leaves(self):
        """trace_model convenience function honours module_leaves."""

        class LeafModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = LeafModule()

            def forward(self, x):
                return self.leaf(x) + 1

        model = OuterModule()
        graph = trace_model(
            model,
            (torch.randn(2, 4),),
            module_leaves={LeafModule: lambda m, module_qualified_name=None: True},
        )
        graph.lint()

        leaf_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is model.leaf
        ]
        self.assertEqual(len(leaf_nodes), 1)

    def test_trace_inplace_setitem_ellipsis_1(self):
        """Traces copy[..., index] = update via new_zeros and verifies graph structure."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = torch.zeros((1, 8192, 4), dtype=torch.float32)

            def forward(self, index, update):
                copy = update.new_zeros(self.params.shape)
                copy[..., index] = update
                return copy

        model = Model()
        index = torch.tensor([0, 3, 2, 1], dtype=torch.int64)
        update = (torch.arange(4 * 8192) + 10).reshape((-1, 4)).to(torch.float32)

        tracer = GraphTracer()
        graph = tracer.trace(
            model,
            args=(index, update),
            dynamic_shapes={"index": {0: "batch"}, "update": {0: "batch"}},
        )
        graph.lint()

        # There must be an operator.setitem node in the graph.
        setitem_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is operator.setitem
        ]
        self.assertEqual(len(setitem_nodes), 1, "Expected exactly one operator.setitem node")

        # The setitem indices must be a tuple (Ellipsis, <FX node for index>).
        si_node = setitem_nodes[0]
        si_indices = si_node.args[1]
        self.assertIsInstance(si_indices, tuple, "setitem indices must be a tuple")
        self.assertEqual(len(si_indices), 2, "setitem indices tuple must have 2 elements")
        self.assertIs(si_indices[0], Ellipsis, "First index must be Ellipsis")
        self.assertIsInstance(
            si_indices[1],
            torch.fx.Node,
            "Second index must be an FX Node (unwrapped TracingTensor)",
        )

        # The output must reference the setitem node, not the original new_zeros node.
        output_node = next(n for n in graph.nodes if n.op == "output")
        result_node = output_node.args[0]
        self.assertIs(
            result_node,
            si_node,
            "Output must reference the setitem node, not the original new_zeros",
        )

    def test_trace_inplace_setitem_ellipsis_2(self):
        """Traces copy[..., index] = update via self.params.clone() and verifies graph."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = torch.zeros((1, 8192, 6), dtype=torch.float32)

            def forward(self, index, update):
                copy = self.params.clone()
                copy[..., index] = update
                return copy

        model = Model()
        index = torch.tensor([0, 3, 2, 5], dtype=torch.int64)
        update = (torch.arange(4 * 8192) + 10).reshape((-1, 4)).to(torch.float32)

        tracer = GraphTracer()
        graph = tracer.trace(
            model,
            args=(index, update),
            dynamic_shapes={"index": {0: "batch"}, "update": {0: "batch"}},
        )
        graph.lint()

        # The module attribute must be restored after tracing.
        self.assertIsInstance(
            model.params,
            torch.Tensor,
            "Module attribute must be restored to a plain Tensor after tracing",
        )
        self.assertFalse(
            hasattr(model.params, "_tracer"),
            "Module attribute must not be a TracingTensor after tracing",
        )

        # There must be a clone node in the graph (for self.params.clone()).
        clone_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.clone.default
        ]
        self.assertEqual(len(clone_nodes), 1, "Expected exactly one aten.clone node")

        # There must be an operator.setitem node in the graph.
        setitem_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is operator.setitem
        ]
        self.assertEqual(len(setitem_nodes), 1, "Expected exactly one operator.setitem node")

        # The setitem indices must be a tuple (Ellipsis, <FX node for index>).
        si_node = setitem_nodes[0]
        si_indices = si_node.args[1]
        self.assertIsInstance(si_indices, tuple, "setitem indices must be a tuple")
        self.assertEqual(len(si_indices), 2, "setitem indices tuple must have 2 elements")
        self.assertIs(si_indices[0], Ellipsis, "First index must be Ellipsis")
        self.assertIsInstance(
            si_indices[1],
            torch.fx.Node,
            "Second index must be an FX Node (unwrapped TracingTensor)",
        )

        # The clone node must feed the setitem node.
        clone_node = clone_nodes[0]
        self.assertIs(si_node.args[0], clone_node, "setitem must operate on the clone result")

        # The output must reference the setitem node.
        output_node = next(n for n in graph.nodes if n.op == "output")
        result_node = output_node.args[0]
        self.assertIs(
            result_node, si_node, "Output must reference the setitem node, not the clone node"
        )


class TestNewTracingDynamicCache(ExtTestCase):
    """Tests for GraphTracer tracing models with DynamicCache inputs/outputs."""

    @staticmethod
    def _make_cache(bsize, nheads, slen, dim):
        from yobx.torch.in_transformers.cache_helper import make_dynamic_cache

        return make_dynamic_cache(
            [
                (torch.rand((bsize, nheads, slen, dim)), torch.rand((bsize, nheads, slen, dim))),
                (torch.rand((bsize, nheads, slen, dim)), torch.rand((bsize, nheads, slen, dim))),
            ]
        )

    def test_trace_dynamic_cache_input_static(self):
        """GraphTracer traces DynamicCacheInput with static shapes."""
        from yobx.torch import register_flattening_functions

        try:
            import transformers  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("transformers not installed")
        from yobx.torch.testing._model_eval_cases import DynamicCacheInput

        _bsize, _nheads, _slen, _dim = 2, 4, 3, 7
        model = DynamicCacheInput()
        inputs = (
            torch.rand((_bsize, _nheads, _slen, _dim)),
            self._make_cache(_bsize, _nheads, _slen, _dim),
        )
        # GraphTracer must be created before register_flattening_functions to
        # avoid an optree re-registration conflict on some environments.
        tracer = GraphTracer()
        with register_flattening_functions(patch_transformers=True):
            graph = tracer.trace(model, inputs)
        graph.lint()

        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        # x + 4 tensors from the 2-layer DynamicCache (key0, val0, key1, val1)
        self.assertEqual(len(ph_nodes), 5)

        output_node = next(n for n in graph.nodes if n.op == "output")
        output_val = output_node.args[0]
        # Output is flattened: x + 4 mean-reduced cache tensors
        self.assertIsInstance(output_val, tuple)
        self.assertEqual(len(output_val), 5)

        call_nodes = [n for n in graph.nodes if n.op == "call_function"]
        self.assertEqual(len(call_nodes), 4, "Expected 4 mean.dim nodes (one per cache tensor)")

    def test_trace_dynamic_cache_input_dynamic(self):
        """GraphTracer traces DynamicCacheInput with dynamic shapes."""
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("transformers not installed")
        from yobx.torch import register_flattening_functions
        from yobx.torch.testing._model_eval_cases import DynamicCacheInput

        _bsize, _nheads, _slen, _dim = 2, 4, 3, 7
        model = DynamicCacheInput()
        inputs = (
            torch.rand((_bsize, _nheads, _slen, _dim)),
            self._make_cache(_bsize, _nheads, _slen, _dim),
        )
        DYN = torch.export.Dim.DYNAMIC
        dynamic_shapes = {"x": {0: DYN, 2: DYN}, "cache": [{0: DYN, 2: DYN}] * 4}

        tracer = GraphTracer()
        with register_flattening_functions(patch_transformers=True):
            graph = tracer.trace(model, inputs, dynamic_shapes=dynamic_shapes)
        graph.lint()

        ph_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(ph_nodes), 5)
        ph_names = [n.name for n in ph_nodes]
        self.assertIn("x", ph_names)
        # cache tensors should be named cache_0 … cache_3
        for i in range(4):
            self.assertIn(f"cache_{i}", ph_names)

        output_node = next(n for n in graph.nodes if n.op == "output")
        output_val = output_node.args[0]
        self.assertIsInstance(output_val, tuple)
        self.assertEqual(len(output_val), 5)


class TestGraphTracerTorchCheck(ExtTestCase):
    """Tests for torch._check interception in GraphTracer."""

    def test_handle_check_tracing_bool_registers_condition(self):
        """_handle_check registers a TracingBool condition so it resolves to True."""
        if not hasattr(torch, "_check"):
            return
        from yobx.torch.new_tracing.tracer import GraphTracer
        from yobx.torch.new_tracing.shape import (
            TracingInt,
            TracingBool,
            clear_conditions,
            _known_true_conditions,
        )

        clear_conditions()
        tracer = GraphTracer()
        tb = TracingInt("batch") > 0
        self.assertIsInstance(tb, TracingBool)

        tracer._handle_check(tb)
        self.assertIn(tb.value, _known_true_conditions)
        clear_conditions()

    def test_handle_check_concrete_true_noop(self):
        """_handle_check silently accepts concrete True."""
        tracer = GraphTracer()
        # Must not raise.
        tracer._handle_check(True)

    def test_handle_check_concrete_false_noop(self):
        """_handle_check silently accepts concrete False (symbolic dim stored as 0)."""
        tracer = GraphTracer()
        # Must not raise even for False – symbolic dims are stored as 0, making
        # conditions like 0>0 evaluate to False at trace time.
        tracer._handle_check(False)

    def test_trace_with_torch_check_dynamic_shape(self):
        """Tracing a model that calls torch._check on a dynamic shape succeeds.

        The assertion uses a symbolic dim; the condition should be registered and
        any subsequent use of the same condition resolves to True.
        """
        if not hasattr(torch, "_check"):
            return
        from yobx.torch.new_tracing.tracer import GraphTracer
        from yobx.torch.new_tracing.shape import clear_conditions

        class ModelWithCheck(torch.nn.Module):
            def forward(self, x):
                # x.shape[0] is TracingInt("batch") during tracing.
                torch._check(x.shape[0] > 0)
                return x + 1

        clear_conditions()
        model = ModelWithCheck()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(4, 8),), dynamic_shapes={"x": {0: "batch"}})
        graph.lint()
        clear_conditions()

    def test_trace_with_torch_check_condition_resolves_if(self):
        """torch._check registers a condition that resolves a later if-guard.

        The model calls torch._check(x.shape[0] > 0) and then uses
        ``if x.shape[0] > 0:`` as a guard.  During tracing the condition should
        resolve to True (via the registry) so the true branch is taken.
        """
        if not hasattr(torch, "_check"):
            return
        from yobx.torch.new_tracing.tracer import GraphTracer
        from yobx.torch.new_tracing.shape import clear_conditions

        branch_taken = []

        class ModelWithGuard(torch.nn.Module):
            def forward(self, x):
                torch._check(x.shape[0] > 0)
                if x.shape[0] > 0:
                    branch_taken.append("true")
                    return x * 2
                else:
                    branch_taken.append("false")
                    return x + 1

        clear_conditions()
        model = ModelWithGuard()
        tracer = GraphTracer()
        graph = tracer.trace(model, (torch.randn(4, 8),), dynamic_shapes={"x": {0: "batch"}})
        graph.lint()
        self.assertIn("true", branch_taken, "The true branch should have been taken")
        clear_conditions()

    def test_trace_cond_numel_predicate(self):
        """torch.cond with a numel()>0 predicate emits proper FX nodes.

        Replicates the ControlFlowCondNonZero model: the predicate is
        ``image_features.numel() > 0``.  During tracing this must produce
        ``aten.sym_size.int`` + ``aten.mul.Tensor`` + ``aten.gt.Scalar``
        nodes rather than a raw Python :class:`~yobx.torch.new_tracing.shape.TracingBool`,
        so the ONNX ``If`` node receives a proper boolean tensor input.
        """

        class CondNonZeroModel(torch.nn.Module):
            def forward(self, input_ids, image_features, vocab_size):
                def then_branch(input_ids, image_features, vocab_size):
                    input_shape = input_ids.size()
                    input_ids = input_ids.view(-1, input_shape[-1])
                    condition = (input_ids < 0) & (input_ids > -int(1e9))
                    positions = torch.nonzero(condition, as_tuple=True)
                    input_ids = input_ids.clamp_min(0).clamp_max(vocab_size)
                    return (input_ids, positions[0], positions[1])

                def else_branch(input_ids, image_features, vocab_size):
                    r = torch.where(torch.zeros((1, 1), dtype=torch.bool))
                    return (input_ids, r[0], r[1])

                a, b, c = torch.cond(
                    image_features.numel() > 0,
                    then_branch,
                    else_branch,
                    [input_ids, image_features, vocab_size],
                )
                return a, b, c

        model = CondNonZeroModel()
        inputs = (
            (torch.arange(24) - 8).reshape((2, -1)).to(torch.int64),
            torch.arange(32).reshape((2, -1)).to(torch.float32),
            1025,
        )
        DIM = torch.export.Dim
        dynamic_shapes = ({0: DIM("batch")}, {0: DIM("batch"), 1: DIM("seq_length")}, None)
        tracer = GraphTracer()
        graph = tracer.trace(model, inputs, dynamic_shapes=dynamic_shapes)
        graph.lint()

        # The predicate chain: sym_size.int × 2  →  mul.Tensor  →  gt.Scalar
        sym_size_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.aten.sym_size.int  # type: ignore[attr-defined]
        ]
        self.assertGreaterEqual(len(sym_size_nodes), 2, "Expected sym_size.int nodes for numel()")

        mul_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.mul.Tensor
        ]
        self.assertGreaterEqual(len(mul_nodes), 1, "Expected mul.Tensor node for numel product")

        gt_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.gt.Scalar
        ]
        self.assertEqual(len(gt_nodes), 1, "Expected exactly one gt.Scalar node for numel() > 0")

        # The cond node must receive the gt.Scalar node as its predicate arg.
        cond_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is torch.cond
        ]
        self.assertEqual(len(cond_nodes), 1, "Expected exactly one torch.cond node")
        pred_arg = cond_nodes[0].args[0]
        self.assertIs(
            pred_arg,
            gt_nodes[0],
            "torch.cond predicate must be the gt.Scalar FX node, not a Python TracingBool",
        )

        # Sub-graphs must also be valid.
        for _, sub in tracer._sub_tracers.items():
            sub.graph.lint()

        # ------------------------------------------------------------------
        # Convert to ONNX with new-tracing and verify the result is valid
        # and produces numerically correct outputs when run with OnnxRuntime.
        # ------------------------------------------------------------------
        import onnx
        from yobx.torch import ExportOptions
        from yobx.torch.export_options import TracingMode
        from yobx.torch.interpreter import to_onnx

        artifact = to_onnx(
            model,
            inputs,
            dynamic_shapes=dynamic_shapes,
            export_options=ExportOptions(tracing=TracingMode.NEW_TRACING),
        )
        onx = artifact.proto
        self.assertIsInstance(onx, onnx.ModelProto, "to_onnx must return a valid ModelProto")
        # The ONNX model has 2 tensor inputs (vocab_size is embedded as a constant).
        self.assertEqual(
            len(onx.graph.input), 2, f"Expected 2 ONNX graph inputs, got {len(onx.graph.input)}"
        )
        self.assertEqual(len(onx.graph.output), 3, "Expected 3 ONNX graph outputs")

        # Verify numerical correctness with OnnxRuntime.
        try:
            import onnxruntime
        except ImportError:
            return

        import numpy as np

        expected_a, expected_b, expected_c = model(*inputs)
        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        inp_names = [i.name for i in sess.get_inputs()]
        tensor_inputs = [inp for inp in inputs if isinstance(inp, torch.Tensor)]
        feeds = {name: t.numpy() for name, t in zip(inp_names, tensor_inputs)}
        got_a, got_b, got_c = sess.run(None, feeds)
        self.assertTrue(
            np.array_equal(expected_a.numpy(), got_a),
            "ONNX output 0 (input_ids) must match eager",
        )
        self.assertTrue(
            np.array_equal(expected_b.numpy(), got_b),
            "ONNX output 1 (positions[0]) must match eager",
        )
        self.assertTrue(
            np.array_equal(expected_c.numpy(), got_c),
            "ONNX output 2 (positions[1]) must match eager",
        )

    def test_trace_shape_as_index(self):
        """Tracing a model that uses y.shape[1] as a slice endpoint.

        Replicates the SignatureShapeAsIndex pattern:
        ``return t[:, :y.shape[1]]`` where dimension 1 of *y* is dynamic.
        The tracer must emit ``aten.sym_size.int(y, 1)`` and use that node as
        the ``end`` argument of ``aten.slice.Tensor`` rather than falling back
        to the constant ``0``.
        """

        class ShapeAsIndexModel(torch.nn.Module):
            def forward(self, x, y):
                return x[:, : y.shape[1]]

        model = ShapeAsIndexModel()
        x = torch.randn(4, 5)
        y = torch.randn(4, 3)
        DIM = torch.export.Dim
        dynamic_shapes = {"x": {0: DIM("batch")}, "y": {0: DIM("batch"), 1: DIM("length")}}
        tracer = GraphTracer()
        graph = tracer.trace(model, (x, y), dynamic_shapes=dynamic_shapes)
        graph.lint()

        # There must be an aten.sym_size.int node for dimension 1 of y.
        sym_size_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.aten.sym_size.int  # type: ignore[attr-defined]
        ]
        self.assertGreaterEqual(
            len(sym_size_nodes), 1, "Expected at least one aten.sym_size.int node"
        )

        # There must be an aten.slice.Tensor node.
        slice_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.slice.Tensor
        ]
        self.assertGreaterEqual(
            len(slice_nodes), 1, "Expected at least one aten.slice.Tensor node"
        )

        # The sym_size.int node must feed into the slice.Tensor node as an arg.
        sym_size_node = sym_size_nodes[0]
        slice_with_sym = [n for n in slice_nodes if sym_size_node in n.args]
        self.assertGreaterEqual(
            len(slice_with_sym),
            1,
            "aten.slice.Tensor must receive the aten.sym_size.int node as an argument",
        )

        # The output shape's dimension 1 must be symbolic.
        from yobx.torch.new_tracing.shape import TracingInt

        output_node = next(n for n in graph.nodes if n.op == "output")
        result_val = output_node.args[0]
        if isinstance(result_val, torch.fx.Node):
            result_shape = result_val.meta.get("val")
            if result_shape is not None and hasattr(result_shape, "_tracing_shape"):
                dim1 = result_shape._tracing_shape.dims[1]
                self.assertIsInstance(
                    dim1, TracingInt, "Output dimension 1 must be a TracingInt (symbolic)"
                )
                self.assertFalse(
                    dim1.is_static, "Output dimension 1 must be symbolic, not static"
                )

    def test_trace_cat_ndim_indirect(self):
        """Tracing a model that cats two tensors then checks ndim.

        Replicates the ControlFlowIndirectRanksCat pattern:
        ``cat = torch.cat([x1, y1], dim=1); if cat.ndim == 2: return cat.clone()``.

        Since ndim is always known statically, the true branch must be taken and
        the graph must contain aten.cat.default and aten.clone.default nodes.
        The cat output shape must carry a compound symbolic dimension (e.g. ``seq+4``)
        for the concatenated axis.
        """
        from yobx.torch.new_tracing.shape import TracingInt

        class CatNdimModel(torch.nn.Module):
            def forward(self, x, y):
                x1 = x + 1
                y1 = y + 2
                cat = torch.cat([x1, y1], dim=1)
                if cat.ndim == 2:
                    return cat.clone()
                return cat / cat.ndim

        DIM = torch.export.Dim
        model = CatNdimModel()
        x = torch.rand(3, 4)
        y = torch.rand(3, 2)
        dynamic_shapes = {"x": {0: DIM("batch")}, "y": {0: DIM("batch"), 1: DIM("seq")}}
        tracer = GraphTracer()
        graph = tracer.trace(model, (x, y), dynamic_shapes=dynamic_shapes)
        graph.lint()

        call_targets = [n.target for n in graph.nodes if n.op == "call_function"]

        # aten.cat.default must appear
        self.assertIn(
            torch.ops.aten.cat.default,
            call_targets,
            "Expected aten.cat.default node in traced graph",
        )
        # True branch (clone) must be taken since ndim == 2 is always True
        self.assertIn(
            torch.ops.aten.clone.default,
            call_targets,
            "Expected aten.clone.default node (true branch) in traced graph",
        )

        # The cat node's output shape must have a compound symbolic dimension
        cat_node = next(
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.cat.default
        )
        cat_shape = cat_node.meta["val"]._tracing_shape
        dim1 = cat_shape.dims[1]
        self.assertIsInstance(
            dim1, TracingInt, "Cat output dim 1 must be a TracingInt (symbolic expression)"
        )
        self.assertFalse(dim1.is_static, "Cat output dim 1 must be symbolic, not static")

    def test_can_prove_expr_nonzero_helper(self):
        """_can_prove_expr_nonzero returns True when expression is provably non-zero."""
        from yobx.torch.new_tracing.shape import (
            _can_prove_expr_nonzero,
            clear_conditions,
            register_condition,
            TracingInt,
        )

        clear_conditions()
        # Register two individual positivity constraints.
        register_condition(TracingInt("a") > 0)
        register_condition(TracingInt("b") > 0)

        self.assertTrue(_can_prove_expr_nonzero("a"), "single positive var should be provable")
        self.assertTrue(_can_prove_expr_nonzero("5"), "positive integer constant is provable")
        self.assertTrue(_can_prove_expr_nonzero("5*a*b"), "product of known-positive factors")
        self.assertTrue(_can_prove_expr_nonzero("a*b"), "product of two known-positive vars")
        # Sum of positive vars is also provably nonzero via evaluate_expression.
        self.assertTrue(_can_prove_expr_nonzero("a+b"), "sum of known-positive vars is nonzero")
        self.assertFalse(
            _can_prove_expr_nonzero("c"), "unknown variable c should not be provable"
        )
        self.assertFalse(
            _can_prove_expr_nonzero("a*c"), "product with unknown var should not be provable"
        )
        self.assertFalse(
            _can_prove_expr_nonzero("0"), "zero constant should not be provable nonzero"
        )
        # a - b evaluates to 0 when both are mapped to 1, so cannot be proven nonzero.
        self.assertFalse(_can_prove_expr_nonzero("a-b"), "difference might be zero, not provable")
        clear_conditions()

    def test_tracing_bool_can_prove_nonzero_resolves_to_false(self):
        """TracingBool('E==0') resolves to False when E is a provably nonzero product.

        This covers the ControlFlowNumelZero3 pattern:
        ``torch._check(x.shape[0] > 0); torch._check(x.shape[2] > 0)``
        followed by ``if x.numel() == 0:``.  The numel expression is a product
        of positive factors so the condition should resolve to False.
        """
        from yobx.torch.new_tracing.shape import (
            TracingBool,
            TracingInt,
            clear_conditions,
            register_condition,
        )

        clear_conditions()
        # Simulate two separate torch._check calls on individual dimensions.
        register_condition(TracingInt("d0") > 0)
        register_condition(TracingInt("d2") > 0)

        # Numel expression: 2 * 5 * d0 * d2 = 10*d0*d2
        cond = TracingBool("10*d0*d2==0")
        self.assertFalse(bool(cond), "10*d0*d2==0 must be False since d0>0 and d2>0")

        # A simple single-var case.
        cond2 = TracingBool("d0==0")
        self.assertFalse(bool(cond2), "d0==0 must be False since d0>0")

        # An unknown variable should still raise.
        cond3 = TracingBool("unknown==0")
        with self.assertRaises(ValueError):
            bool(cond3)
        clear_conditions()

    def test_trace_controlflow_numel_zero_3(self):
        """Tracing ControlFlowNumelZero3 with two separate torch._check calls succeeds.

        The model uses:
            torch._check(x.shape[0] > 0)
            torch._check(x.shape[2] > 0)
            if x.numel() == 0: return 0
            return x.shape[-2]

        The condition ``x.numel() == 0`` must be resolvable to False via the
        two registered positivity constraints, so the true-branch (``return 0``)
        is not taken and tracing succeeds.
        """
        if not hasattr(torch, "_check"):
            return
        from yobx.torch.new_tracing.shape import clear_conditions

        class ControlFlowNumelZero3(torch.nn.Module):
            def forward(self, x):
                def empty_cache(x):
                    torch._check(x.shape[0] > 0)
                    torch._check(x.shape[2] > 0)
                    if x.numel() == 0:
                        return 0
                    return x.shape[-2]

                size = (empty_cache(x), 1)
                return torch.full(size, fill_value=2)

        clear_conditions()
        model = ControlFlowNumelZero3()
        x = torch.rand(3, 2, 2, 5)
        tracer = GraphTracer()
        graph = tracer.trace(
            model,
            (x,),
            dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC, 2: torch.export.Dim.DYNAMIC}},
        )
        graph.lint()
        clear_conditions()

        # The graph must contain a torch.full call (the return value).
        full_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is torch.full
        ]
        self.assertEqual(len(full_nodes), 1, "Expected exactly one torch.full node")

    def test_is_positivity_condition_helper(self):
        """_is_positivity_condition returns True only for simple >0 / >=N constraints."""
        from yobx.torch.new_tracing.shape import _is_positivity_condition

        # Simple "var>0" forms.
        self.assertTrue(_is_positivity_condition("d0>0"))
        self.assertTrue(_is_positivity_condition("_dyn_1>0"))
        # "var>=N" forms where N >= 1.
        self.assertTrue(_is_positivity_condition("d0>=1"))
        self.assertTrue(_is_positivity_condition("d0>=2"))
        # Non-positivity conditions.
        self.assertFalse(_is_positivity_condition("d0>0 and d2>0"))
        self.assertFalse(_is_positivity_condition("d0!=0"))
        self.assertFalse(_is_positivity_condition("d0==0"))
        self.assertFalse(_is_positivity_condition("d0>=0"))
        self.assertFalse(_is_positivity_condition("unknown==0"))
        self.assertFalse(_is_positivity_condition(""))

    def test_tracing_bool_positivity_condition_self_registers(self):
        """TracingBool('var>0') self-registers as known-True when bool() is called.

        This covers the ControlFlowNumelZero4 pattern where Python's ``and``
        operator evaluates ``bool(TracingBool("d0>0"))`` before the full compound
        expression reaches ``torch._check``.  The condition must be registered so
        that subsequent ``if x.numel() == 0:`` guards can be resolved to False.
        """
        from yobx.torch.new_tracing.shape import (
            TracingBool,
            TracingInt,
            _known_true_conditions,
            clear_conditions,
        )

        clear_conditions()
        # Simulates what happens when Python evaluates
        # `torch._check(x.shape[0] > 0 and x.shape[2] > 0)`:
        # The `and` calls bool() on the left operand.
        cond = TracingBool("d0>0")
        self.assertNotIn("d0>0", _known_true_conditions, "should not be registered yet")
        result = bool(cond)
        self.assertTrue(result, "positivity condition must resolve to True")
        self.assertIn("d0>0", _known_true_conditions, "must be self-registered after bool()")

        # After self-registration the numel==0 condition for a product involving
        # d0 must now resolve to False.
        # Also register d2>0 to simulate the second `and` operand being registered.
        from yobx.torch.new_tracing.shape import register_condition

        register_condition(TracingInt("d2") > 0)
        cond_numel = TracingBool("10*d0*d2==0")
        self.assertFalse(
            bool(cond_numel), "10*d0*d2==0 must be False after registering both dims"
        )
        clear_conditions()

    def test_trace_controlflow_numel_zero_4(self):
        """Tracing ControlFlowNumelZero4 with a combined torch._check succeeds.

        The model uses a single combined check:
            torch._check(x.shape[0] > 0 and x.shape[2] > 0)
            if x.numel() == 0: return 0
            return x.shape[-2]

        Python's ``and`` operator calls ``bool()`` on the left ``TracingBool``
        before ``torch._check`` is called.  The self-registration mechanism in
        :meth:`TracingBool.__bool__` must register both constraints so that
        ``if x.numel() == 0:`` resolves to False and tracing succeeds.
        """
        if not hasattr(torch, "_check"):
            return
        from yobx.torch.new_tracing.shape import clear_conditions

        class ControlFlowNumelZero4(torch.nn.Module):
            def forward(self, x):
                def empty_cache(x):
                    torch._check(x.shape[0] > 0 and x.shape[2] > 0)
                    if x.numel() == 0:
                        return 0
                    return x.shape[-2]

                size = (empty_cache(x), 1)
                return torch.full(size, fill_value=2)

        clear_conditions()
        model = ControlFlowNumelZero4()
        x = torch.rand(3, 2, 2, 5)
        tracer = GraphTracer()
        graph = tracer.trace(
            model,
            (x,),
            dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC, 2: torch.export.Dim.DYNAMIC}},
        )
        graph.lint()
        clear_conditions()

        # The graph must contain a torch.full call (the return value).
        full_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is torch.full
        ]
        self.assertEqual(len(full_nodes), 1, "Expected exactly one torch.full node")

    def test_trace_controlflow_numel_zero_5(self):
        """Tracing ControlFlowNumelZero5 succeeds.

        The model uses ``torch._check(x.numel() != 0)`` followed by a
        per-dimension ``!= 0`` guard::

            torch._check(x.numel() != 0)
            if x.shape[0] != 0 and x.shape[2] != 0:
                return 0
            return x.shape[-2]

        Under the assertion ``numel != 0`` (which registers
        ``"10*d0*d2!=0"`` as a known-true condition), each individual
        dimension ``shape[0] != 0`` and ``shape[2] != 0`` can be proved
        True because the respective symbolic name is a multiplicative
        factor of the known-nonzero product.

        The true branch ``return 0`` is therefore taken, producing a
        static size ``(0, 1)`` for ``torch.full``.  The tracer must still
        emit an FX node for ``torch.full`` with the concrete size so that
        the output is a :class:`~yobx.torch.new_tracing.tensor.TracingTensor`.
        """
        if not hasattr(torch, "_check"):
            return
        from yobx.torch.new_tracing.shape import clear_conditions

        class ControlFlowNumelZero5(torch.nn.Module):
            def forward(self, x):
                def empty_cache(x):
                    torch._check(x.numel() != 0)
                    if x.shape[0] != 0 and x.shape[2] != 0:
                        return 0
                    return x.shape[-2]

                size = (empty_cache(x), 1)
                return torch.full(size, fill_value=2)

        clear_conditions()
        model = ControlFlowNumelZero5()
        x = torch.rand(3, 2, 2, 5)
        tracer = GraphTracer()
        graph = tracer.trace(
            model,
            (x,),
            dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC, 2: torch.export.Dim.DYNAMIC}},
        )
        graph.lint()
        clear_conditions()

        # The graph must contain a torch.full call (the return value).
        full_nodes = [
            n for n in graph.nodes if n.op == "call_function" and n.target is torch.full
        ]
        self.assertEqual(len(full_nodes), 1, "Expected exactly one torch.full node")

    def test_can_prove_expr_nonzero_from_neq_conditions(self):
        """_can_prove_expr_nonzero_from_neq_conditions returns True for factors of P!=0.

        When ``torch._check(x.numel() != 0)`` registers ``"10*d0*d2!=0"``,
        the individual factors ``"d0"`` and ``"d2"`` must each be proved
        nonzero by :func:`_can_prove_expr_nonzero_from_neq_conditions`.
        """
        from yobx.torch.new_tracing.shape import (
            TracingInt,
            _can_prove_expr_nonzero_from_neq_conditions,
            clear_conditions,
            register_condition,
        )

        clear_conditions()

        # Simulate numel registration: 10*d0*d1 != 0
        d0 = TracingInt("_dyn_0")
        d1 = TracingInt(2)
        d2 = TracingInt("_dyn_1")
        d3 = TracingInt(5)
        numel = d0 * d1 * d2 * d3
        register_condition(numel != 0)

        self.assertTrue(
            _can_prove_expr_nonzero_from_neq_conditions("_dyn_0"),
            "_dyn_0 must be provably nonzero as a factor of 10*_dyn_0*_dyn_1",
        )
        self.assertTrue(
            _can_prove_expr_nonzero_from_neq_conditions("_dyn_1"),
            "_dyn_1 must be provably nonzero as a factor of 10*_dyn_0*_dyn_1",
        )
        self.assertFalse(
            _can_prove_expr_nonzero_from_neq_conditions("_dyn_2"),
            "_dyn_2 is not a factor of the registered condition",
        )
        clear_conditions()


if __name__ == "__main__":
    unittest.main(verbosity=2)
