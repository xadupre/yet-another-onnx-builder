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


if __name__ == "__main__":
    unittest.main(verbosity=2)
