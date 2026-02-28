import copy
import operator
import unittest
import torch
from yobx.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_transformers,
    hide_stdout,
    skipif_ci_windows,
)
from yobx.torch.tracing import (
    CustomTracer,
    CustomProxy,
    CustomParameterProxy,
    CondCCOp,
    _len,
    _isinstance,
    replace_problematic_function_before_tracing,
    setitem_with_transformation,
    tree_unflatten_with_proxy,
)
from yobx.torch import register_flattening_functions
from yobx.torch.transformers.cache_helper import make_dynamic_cache


@requires_torch("2.0")
class TestCustomTracer(ExtTestCase):
    def test_import(self):
        self.assertIsNotNone(CustomTracer)
        self.assertIsNotNone(CustomProxy)

    def test_import_from_package(self):
        from yobx.torch import CustomTracer as CT, CustomProxy as CP

        self.assertIsNotNone(CT)
        self.assertIsNotNone(CP)

    def test_trace_simple_add(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        graph = CustomTracer().trace(Model())
        ops = [n.op for n in graph.nodes]
        self.assertIn("placeholder", ops)
        self.assertIn("output", ops)

    def test_trace_linear(self):
        model = torch.nn.Linear(4, 4)
        graph = CustomTracer().trace(model)
        self.assertIsNotNone(graph)
        node_ops = {n.op for n in graph.nodes}
        self.assertIn("placeholder", node_ops)
        self.assertIn("output", node_ops)

    def test_trace_inplace_add(self):
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
        tracer = CustomTracer()
        # Standard nn.Linear is a leaf by default
        linear = torch.nn.Linear(4, 4)
        self.assertTrue(tracer.is_leaf_module(linear, "linear"))

    def test_is_leaf_module_custom(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x

        tracer = CustomTracer(module_leaves={MyModule: lambda m, module_qualified_name: True})
        self.assertTrue(tracer.is_leaf_module(MyModule(), "mymodule"))

    def test_len_with_proxy(self):
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
        self.assertEqual(_len([1, 2, 3]), 3)

    def test_isinstance_plain(self):
        self.assertTrue(_isinstance([1, 2], list))
        self.assertFalse(_isinstance((1, 2), list))

    def test_replace_problematic_function(self):
        original_cat = torch.cat
        with replace_problematic_function_before_tracing():
            # Inside context: torch.cat is replaced
            self.assertIsNot(torch.cat, original_cat)
        # After context: torch.cat is restored
        self.assertIs(torch.cat, original_cat)

    def test_cond_cc_op(self):
        op = CondCCOp()
        self.assertIsNotNone(op)
        self.assertIsInstance(op, torch._ops.HigherOrderOperator)

    def test_trace_with_module_leaves(self):
        class LeafModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = LeafModule()

            def forward(self, x):
                return self.leaf(x) + 1

        tracer = CustomTracer(module_leaves={LeafModule: lambda m, module_qualified_name: True})
        graph = tracer.trace(OuterModule())
        # LeafModule should appear as call_module node
        module_calls = [n for n in graph.nodes if n.op == "call_module"]
        self.assertEqual(len(module_calls), 1)

    def test_create_arg_types(self):
        tracer = CustomTracer()
        # Need a root module for context
        model = torch.nn.Linear(4, 4)
        _ = tracer.trace(model)
        self.assertEqual(tracer.create_arg(bool), torch.bool)
        self.assertEqual(tracer.create_arg(int), torch.int64)
        self.assertEqual(tracer.create_arg(float), torch.float32)
        self.assertEqual(tracer.create_arg(complex), torch.complex64)

    def test_remove_inplace_no_inplace(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        tracer = CustomTracer()
        graph = tracer.trace(Model())
        # No inplace to remove
        result = CustomTracer.remove_inplace(graph)
        self.assertEqual(result, 0)

    def test_trace_setitem(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.clone()
                y[0] = 0
                return y

        graph = CustomTracer().trace(Model())
        self.assertIsNotNone(graph)
        graph.lint()

    def test_make_args_names_dict(self):
        import torch.utils._pytree as pytree

        concrete_args = {"x": torch.randn(3, 4), "y": torch.randn(3, 4)}
        flat, _spec = pytree.tree_flatten(concrete_args)
        names = CustomTracer.make_args_names(concrete_args, flat)
        self.assertEqual(names, ["x", "y"])

    def test_make_args_names_list(self):
        import torch.utils._pytree as pytree

        concrete_args = [torch.randn(3, 4), torch.randn(3, 4)]
        flat, _spec = pytree.tree_flatten(concrete_args)
        names = CustomTracer.make_args_names(concrete_args, flat)
        self.assertEqual(names, ["a0", "a1"])

    def test_make_args_names_dict_with_list_value(self):
        import torch.utils._pytree as pytree

        concrete_args = {
            "x": torch.randn(3, 4),
            "items": [torch.randn(2, 4), torch.randn(2, 4)],
        }
        flat, _spec = pytree.tree_flatten(concrete_args)
        names = CustomTracer.make_args_names(concrete_args, flat)
        self.assertEqual(names, ["x", "items_0", "items_1"])

    def test_make_wrapped_model_dict_tensors(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        concrete_args = {"x": torch.randn(3, 4), "y": torch.randn(3, 4)}
        wrapped, arg_names = CustomTracer.make_wrapped_model(model, concrete_args)
        self.assertIsInstance(wrapped, torch.nn.Module)
        self.assertEqual(arg_names, ["x", "y"])
        self.assertTrue(hasattr(wrapped, "_traced_m2"))
        x, y = torch.randn(3, 4), torch.randn(3, 4)
        result = wrapped(x, y)
        self.assertEqual(result.shape, torch.Size([3, 4]))

    def test_make_wrapped_model_dict_tensors_mixed(self):
        class Model(torch.nn.Module):
            def forward(self, x, items):
                return x + items[0] + items[1]

        model = Model()
        concrete_args = {
            "x": torch.randn(3, 4),
            "items": [torch.randn(3, 4), torch.randn(3, 4)],
        }
        wrapped, arg_names = CustomTracer.make_wrapped_model(model, concrete_args)
        self.assertIsInstance(wrapped, torch.nn.Module)
        self.assertEqual(arg_names, ["x", "items_0", "items_1"])
        self.assertTrue(hasattr(wrapped, "_traced_m2"))
        x, i0, i1 = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
        result = wrapped(x, i0, i1)
        self.assertEqual(result.shape, torch.Size([3, 4]))

    @requires_transformers("4.57")
    def test_make_wrapped_model_dynamic_cache(self):
        from yobx.torch import register_flattening_functions

        class ModelWithCache(torch.nn.Module):
            def forward(self, x, cache):
                return x

        model = ModelWithCache()
        cache = make_dynamic_cache([(torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8))])
        concrete_args = {"x": torch.randn(2, 5, 16), "cache": cache}
        with register_flattening_functions(patch_transformers=True):
            wrapped, arg_names = CustomTracer.make_wrapped_model(model, concrete_args)
        self.assertIsInstance(wrapped, torch.nn.Module)
        self.assertTrue(hasattr(wrapped, "_traced_m1"))
        self.assertEqual(arg_names[0], "x")
        x = torch.randn(2, 5, 16)
        key = torch.randn(2, 4, 5, 8)
        val = torch.randn(2, 4, 5, 8)
        result = wrapped(x, key, val)
        self.assertEqual(result.shape, torch.Size([2, 5, 16]))


@requires_torch("2.0")
class TestTracing(ExtTestCase):
    def test_tracing_simple_proxy(self):
        graph = torch.fx.Graph()
        node = graph.create_node("placeholder", "tx", args=(), kwargs={}, name="txn")
        tr = CustomTracer()
        tr.graph = torch.fx.Graph(tracer_cls=CustomTracer)
        x = CustomProxy(node, tr)
        i = _len(x)
        self.assertIsInstance(i, CustomProxy)

    def test_tracing_abs(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.abs(x)

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_add(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                xc += 2
                return xc + 3

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model)
        self.assertNotIn("add_]", str(graph))
        mod = torch.fx.GraphModule(model, graph)
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_add_(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                xc.add_(2)
                return xc + 3

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model)
        self.assertIn("add_]", str(graph))
        mod = torch.fx.GraphModule(model, graph)
        self.assertIn("add_]", str(mod.graph))
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_add_users(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                xc.add_(2)
                return xc + 3

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model, remove_inplace=False)
        self.assertEqual(
            len([node for node in graph.nodes if len(node.users) == 0 and node.op != "output"]),
            1,
        )
        self.assertIn("(%clone, 3)", str(graph))
        graph = CustomTracer().trace(model, remove_inplace=True)
        self.assertEmpty(
            [node for node in graph.nodes if len(node.users) == 0 and node.op != "output"]
        )
        self.assertNotIn("(%clone, 3)", str(graph))
        mod = torch.fx.GraphModule(model, graph)
        self.assertNotIn("(%clone, 3)", str(mod.graph))
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_add_mul_users(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                xc.add_(2)
                xc.add_(5)
                return xc + 3

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model, remove_inplace=False)
        self.assertEqual(
            len([node for node in graph.nodes if len(node.users) == 0 and node.op != "output"]),
            2,
        )
        self.assertIn("(%clone, 3)", str(graph))
        graph = CustomTracer().trace(model, remove_inplace=True)
        self.assertEmpty(
            [node for node in graph.nodes if len(node.users) == 0 and node.op != "output"]
        )
        self.assertNotIn("(%clone, 3)", str(graph))
        mod = torch.fx.GraphModule(model, graph)
        self.assertNotIn("(%clone, 3)", str(mod.graph))
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_setitem(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                y = xc[:, :2] * 2
                xc[:, :2] = y
                return xc + 3

        model = Model()
        x = torch.ones((4, 4))
        expected = model(x)
        graph = CustomTracer().trace(model)
        self.assertIn("operator.setitem", str(graph))
        mod = torch.fx.GraphModule(model, graph)
        got = mod(x)
        self.assertEqualArray(expected, got)

    def test_tracing_isinstance(self):
        class Model(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, lx):
                if _isinstance(lx, list):
                    return torch.sigmoid(self.linear(x)) + lx
                t = lx[0] * lx[1].sum(axis=1, keepdim=True)
                return torch.sigmoid(self.linear(x)) - t

        model = Model()
        self.assertRaise(
            lambda: CustomTracer().trace(model),
            RuntimeError,
            "Unable to know if cls is from type",
        )

    def test_tracing_len(self):
        class Model(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, lx: list):
                t = lx[0] * lx[1].sum(axis=1, keepdim=True)
                llx = _len(lx)
                tn = t / llx
                return torch.sigmoid(self.linear(x)) - tn

        model = Model()
        inputs = (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        )
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        expected = model(*inputs)
        got = mod(*inputs)
        self.assertEqualArray(expected, got)

    def test_tracing_inplace_setitem_ellipsis(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = torch.zeros((1, 8192, 4), dtype=torch.float32)

            def forward(self, index, update):
                copy_ = self.params.clone()
                copy_[..., index] = update
                return copy_

        model = Model()
        inputs = (
            (torch.tensor([0, 3, 2, 1], dtype=torch.int64)),
            (torch.arange(4 * 8192) + 10).reshape((-1, 4)).to(torch.float32),
        )
        expected = model(*inputs)
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_tracing_list_variable_length(self):
        class Model(torch.nn.Module):
            def __init__(self, n_dims: int = 3, n_targets: int = 1):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)
                self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

            def forward(self, x, lx: list):
                t = torch.cat(lx, axis=1).sum(axis=1, keepdim=True)
                return torch.sigmoid(self.linear(x)) - self.buff + t

        model = Model()
        inputs = (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        )
        expected = model(*inputs)
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_tracing_setitem_mask(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                mask = x.to(bool)
                x[mask] = 2
                return x

        inputs = (torch.randn((2, 3, 3)),)
        model = Model()
        expected = model(*inputs)
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("not supported")
    def test_tracing_cond(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x)

                def false_fn(x):
                    return torch.cos(x)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        inputs = (torch.rand(5, 3),)
        model = Model()
        expected = model(*inputs)
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_index_Tensor_copy_0(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                K_33[2:-2, 2:-2, :-1] = sumx[None, :, None]
                K_33[2:-2, 2:-2, -1] = 0.0
                return K_33

        inputs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        model = Model()
        expected = model(*copy.deepcopy(inputs))
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        self.assertIn(operator.setitem, {n.target for n in graph.nodes})
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_index_Tensor_copy_1(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                K_33[2:-2, 2:-2, :] = sumx[None, :, None]
                return K_33

        inputs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        model = Model()
        expected = model(*copy.deepcopy(inputs))
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        self.assertIn(operator.setitem, {n.target for n in graph.nodes})
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_index_Tensor_copy_2(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                K_33[2:-2, 2:-2, :-1] = sumx[None, :, None]
                return torch.abs(K_33)

        inputs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        model = Model()
        expected = model(*copy.deepcopy(inputs))
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        self.assertIn(operator.setitem, {n.target for n in graph.nodes})
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_index_Tensor_copy_3(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                K_33[2:-2, 2:-2, :-1] = sumx[None, :, None]
                e = torch.abs(K_33)
                K_33[2:-2, 2:-2, -1] = 0.0
                return K_33 + e

        inputs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        model = Model()
        expected = model(*copy.deepcopy(inputs))
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        self.assertIn(operator.setitem, {n.target for n in graph.nodes})
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    def test_index_Tensor_copy_exp(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                torch.exp_(K_33[2:-2, 2:-2, :-1])
                return K_33

        inputs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        model = Model()
        expected = model(*copy.deepcopy(inputs))
        self.assertNotEmpty(expected)
        graph = CustomTracer().trace(model)
        self.assertIn(setitem_with_transformation, {n.target for n in graph.nodes})
        mod = torch.fx.GraphModule(model, graph)
        got = mod(*inputs)
        self.assertNotEmpty(got)
        self.assertEqualArray(expected, got)

    @unittest.skip("TODO: fix it")
    def test_tracing_fixed_list_with_none(self):
        class Model(torch.nn.Module):
            def forward(self, lx):
                x = lx[0]
                if lx[1] is not None:
                    x += lx[1]
                if lx[2] is not None:
                    x += lx[2]
                return x

            _inputs = [
                ([torch.rand((4, 4)), torch.rand((4, 4)), None],),
                ([torch.rand((4, 4)), torch.rand((4, 4)), torch.rand((4, 4))],),
            ]

        inputs = Model._inputs
        model = Model()
        graph = CustomTracer().trace(model)
        for inp in inputs:
            expected = model(*inp)
            mod = torch.fx.GraphModule(model, graph)
            got = mod(*inp)
            self.assertEqualArray(expected, got)

    @unittest.skip("TODO: fix it")
    def test_tracing_int_shape(self):
        class Model(torch.nn.Module):
            @staticmethod
            def add_one(dim: int) -> int:
                return dim + 1

            def forward(self, x):
                y = torch.ones((x.shape[0], x.shape[1] + 1))
                return y

            _inputs = [(torch.rand((4, 4)),), (torch.rand((5, 5)),)]
            _dynamic = {"x": {0: torch.export.Dim("dx"), 1: torch.export.Dim("dy")}}

        inputs = Model._inputs
        model = Model()
        graph = CustomTracer().trace(model, dynamic_shapes=Model._dynamic)
        for inp in inputs:
            expected = model(*inp)
            mod = torch.fx.GraphModule(model, graph)
            got = mod(*inp)
            self.assertEqualArray(expected, got)

    @unittest.skip("TODO: fix it")
    def test_tracing_function_int_shape(self):
        class Model(torch.nn.Module):
            @staticmethod
            def add_one(dim: int) -> int:
                return dim + 1

            def forward(self, x):
                dy1 = Model.add_one(x.shape[1])
                y = torch.ones((x.shape[0], dy1))
                return y

            _inputs = [(torch.rand((4, 4)),), (torch.rand((5, 5)),)]
            _dynamic = {"x": {0: torch.export.Dim("dx"), 1: torch.export.Dim("dy")}}

        inputs = Model._inputs
        model = Model()
        graph = CustomTracer().trace(model, dynamic_shapes=Model._dynamic)
        for inp in inputs:
            expected = model(*inp)
            mod = torch.fx.GraphModule(model, graph)
            got = mod(*inp)
            self.assertEqualArray(expected, got)

    def test_lookup_op(self):
        op = torch._library.utils.lookup_op("aten::masked_fill.Scalar")
        self.assertEqual("aten::masked_fill.Scalar", op.name())

    @skipif_ci_windows("does not work on windows")
    @hide_stdout()
    def test_tracing_with_submodule(self):
        self.skipTest("not implemented yet")

        def filter_position_ids(
            patch_attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            boundaries: torch.Tensor,
            num_patches_per_side: int,
        ):
            for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / p_attn_mask[:, 0].sum())
                fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / p_attn_mask[0].sum())

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (
                    bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w
                ).flatten()
                position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids
            return position_ids

        def scan_filter_position_ids(
            patch_attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            boundaries: torch.Tensor,
            num_patches_per_side: int,
        ):
            def body(p_attn_mask, position_ids_row):
                h_len = torch.tensor(1, dtype=p_attn_mask.dtype) / p_attn_mask[:, 0].sum()
                w_len = torch.tensor(1, dtype=p_attn_mask.dtype) / p_attn_mask[0].sum()
                torch._check(h_len.item() > 0)
                fractional_coords_h = torch.arange(
                    torch.tensor(0.0, dtype=p_attn_mask.dtype),
                    torch.tensor(1 - 1e-6, dtype=p_attn_mask.dtype),
                    h_len,
                )
                torch._check(w_len.item() > 0)
                fractional_coords_w = torch.arange(
                    torch.tensor(0.0, dtype=p_attn_mask.dtype),
                    torch.tensor(1 - 1e-6, dtype=p_attn_mask.dtype),
                    w_len,
                )

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (
                    bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w
                ).flatten()

                row = position_ids_row.clone()
                row[p_attn_mask.view(-1)] = pos_ids
                return [row]

            return torch.ops.higher_order.scan(
                body, [], [patch_attention_mask, position_ids], additional_inputs=[]
            )

        class Model(torch.nn.Module):
            def forward(self, patch_attention_mask, position_ids, boundaries):
                res = scan_filter_position_ids(patch_attention_mask, position_ids, boundaries, 32)
                return res[0]

        patch_attention_mask = torch.randint(0, 17, (32, 32, 32)) >= 1
        patch_attention_mask[:, :, :] = True
        position_ids = torch.zeros((32, 1024), dtype=torch.int64)
        boundaries = (torch.arange(33).to(torch.float32) / 33)[1:-1]
        inputs = (patch_attention_mask, position_ids, boundaries)

        model = Model()
        true_expected = filter_position_ids(*(*inputs, 32))
        expected = model(*inputs)
        self.assertEqualArray(true_expected, expected)

        DYN = torch.export.Dim.DYNAMIC
        with register_flattening_functions(patch_torch=True):
            ep = torch.export.export(model, inputs, dynamic_shapes=({0: DYN}, {0: DYN}, {0: DYN}))

        CustomTracer.remove_inplace(ep.graph, recursive=True, verbose=10)

        for node in ep.graph.nodes:
            if node.op == "get_attr":
                init = getattr(node.graph.owning_module, node.target)
                for n in init.graph.nodes:
                    assert (
                        n.op != "call_function"
                        or n.target != torch.ops.aten._assert_scalar.default
                    ), (
                        f"One assert function was not removed n.target={n.target} "
                        f"in node.target={node.target}"
                    )

        got = ep.module()(*inputs)
        self.assertEqualArray(expected, got)

    @requires_torch("2.9.99")
    def test_tree_unflatten_with_proxy_none(self):
        nested = [
            torch.randn((4, 5)),
            [torch.randn((7, 5)), torch.randn((8, 5))],
            {
                "a": torch.randn((14, 5)),
                "b": torch.randn((12, 5)),
                "cl": [torch.randn((11, 5))],
            },
        ]
        flat_list, tree_spec = torch.utils._pytree.tree_flatten(nested)

        self.assertEqual(len(flat_list), 6)
        unflatten = tree_unflatten_with_proxy(tree_spec, flat_list)
        self.assertEqualAny(nested, unflatten)

    @requires_torch("2.9.99")
    def test_tree_unflatten_with_proxy_custom_proxy(self):
        graph = torch.fx.Graph()
        tr = CustomTracer()
        tr.graph = torch.fx.Graph(tracer_cls=CustomTracer)
        cps = []
        for i in range(6):
            node = graph.create_node("placeholder", f"tx{i}", args=(), kwargs={}, name=f"txn{i}")
            cps.append(CustomProxy(node, tr))

        nested = [
            torch.randn((4, 5)),
            [torch.randn((7, 5)), torch.randn((8, 5))],
            {
                "a": torch.randn((14, 5)),
                "b": torch.randn((12, 5)),
                "cl": [torch.randn((11, 5))],
            },
        ]
        flat_list, tree_spec = torch.utils._pytree.tree_flatten(nested)
        self.assertEqual(len(flat_list), 6)
        unflatten = tree_unflatten_with_proxy(tree_spec, cps)

        expected = [cps[0], [cps[1], cps[2]], {"a": cps[3], "b": cps[4], "cl": [cps[5]]}]
        self.assertEqual(len(expected), len(unflatten))
        for a, b in zip(expected, unflatten):
            self.assertEqual(type(a), type(b))
            if isinstance(a, list):
                self.assertEqual(len(a), len(b))
                for aa, bb in zip(a, b):
                    self.assertEqual(type(aa), type(bb))
            elif isinstance(a, dict):
                self.assertEqual(len(a), len(b))
                self.assertEqual(set(a), set(b))
                for k in a:
                    self.assertEqual(type(a[k]), type(b[k]))

    @requires_torch("2.9.99")
    @requires_transformers("4.57")
    def test_tree_unflatten_with_proxy_dynamic_cache(self):
        graph = torch.fx.Graph()
        tr = CustomTracer()
        tr.graph = torch.fx.Graph(tracer_cls=CustomTracer)
        cps = []
        for i in range(7):
            node = graph.create_node("placeholder", f"tx{i}", args=(), kwargs={}, name=f"txn{i}")
            cps.append(CustomProxy(node, tr))

        nested = [
            torch.randn((4, 5)),
            [torch.randn((7, 5)), torch.randn((8, 5))],
            make_dynamic_cache(
                [(torch.randn(2, 32, 30, 96), torch.randn(2, 32, 30, 96)) for i in range(2)]
            ),
        ]
        with register_flattening_functions(patch_transformers=True):
            flat_list, tree_spec = torch.utils._pytree.tree_flatten(nested)
            self.assertEqual(len(flat_list), 7)
            unflatten = tree_unflatten_with_proxy(tree_spec, cps)

            self.assertEqual(len(nested), len(unflatten))
            for a, b in zip(nested, unflatten):
                if isinstance(a, torch.Tensor):
                    self.assertIsInstance(a, torch.Tensor)
                    self.assertIsInstance(b, CustomProxy)
                else:
                    self.assertEqual(type(a), type(b))
                    t = self.string_type(b, with_shape=True)
                    if "DynamicCache" in t:
                        self.assertEqual(
                            (
                                "DynamicCache(key_cache=#2[CustomProxy(cat),"
                                "CustomProxy(cat_2)], "
                                "value_cache=#2[CustomProxy(cat_1),CustomProxy(cat_3)])"
                            ),
                            t,
                        )

    def test_make_args_names_non_dict(self):
        # When concrete_args is not a dict, names are a0, a1, ...
        t = torch.randn(2, 3)
        names = CustomTracer.make_args_names([t, t, t], [t, t, t])
        self.assertEqual(names, ["a0", "a1", "a2"])

    def test_make_args_names_non_dict_empty(self):
        names = CustomTracer.make_args_names([], [])
        self.assertEqual(names, [])

    def test_make_args_names_dict_single_tensors(self):
        # Dict where each value is a single tensor → keys become names
        t1 = torch.randn(2, 3)
        t2 = torch.randn(4, 5)
        concrete_args = {"x": t1, "y": t2}
        flat_concrete_args = [t1, t2]
        names = CustomTracer.make_args_names(concrete_args, flat_concrete_args)
        self.assertEqual(names, ["x", "y"])

    def test_make_args_names_dict_list_of_tensors(self):
        # Dict where each value is a list of tensors → names become key_0, key_1, ...
        t1, t2, t3 = torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)
        concrete_args = {"past": [t1, t2, t3]}
        flat_concrete_args = [t1, t2, t3]
        names = CustomTracer.make_args_names(concrete_args, flat_concrete_args)
        self.assertEqual(names, ["past_0", "past_1", "past_2"])

    def test_make_args_names_dict_mixed(self):
        # Dict with mixed: a single tensor and a list of tensors
        t1, t2, t3 = torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)
        concrete_args = {"x": t1, "past": [t2, t3]}
        flat_concrete_args = [t1, t2, t3]
        names = CustomTracer.make_args_names(concrete_args, flat_concrete_args)
        self.assertEqual(names, ["x", "past_0", "past_1"])

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
        self.assertFalse(f(model.subb, "subb"))
        graph = CustomTracer(module_leaves=module_leaves).trace(model)
        module_nodes = [n for n in graph.nodes if n.op == "call_module"]
        self.assertEqual(len(module_nodes), 2)
        self.assertEqual(module_nodes[0].target, "suba")
        self.assertEqual(module_nodes[1].target, "subb.linear")

    def test_export_submodule(self):
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

        model = Model()
        self.assertTrue(f(model.suba, "suba"))
        self.assertFalse(f(model.subb, "suba"))
        x, y = torch.randn((5, 3)), torch.randn((5, 3))
        ep = torch.export.export(
            model,
            (x, y),
            dynamic_shapes=({0: torch.export.Dim.DYNAMIC}, {0: torch.export.Dim.DYNAMIC}),
        )
        module_nodes = [n for n in ep.graph.nodes if n.op == "call_module"]
        self.assertEqual(len(module_nodes), 0)

    def test_export_and_trace_submodule(self):
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

        graph = CustomTracer(module_leaves=module_leaves).trace(model)
        module_nodes = [n for n in graph.nodes if n.op == "call_module"]
        self.assertEqual(len(module_nodes), 2)
        self.assertEqual(module_nodes[0].target, "suba")
        self.assertEqual(module_nodes[1].target, "subb.linear")


@requires_torch("2.0")
class TestCustomParameterProxy(ExtTestCase):
    def _make_proxy(self, param, name="weight"):
        graph = torch.fx.Graph()
        node = graph.create_node("get_attr", name, args=(), kwargs={}, name=name)
        tracer = CustomTracer()
        tracer.graph = torch.fx.Graph(tracer_cls=CustomTracer)
        return CustomParameterProxy(tracer, node, name, param)

    def test_custom_parameter_proxy_repr(self):
        param = torch.nn.Parameter(torch.randn(3, 4))
        proxy = self._make_proxy(param, name="weight")
        self.assertEqual(repr(proxy), "CustomParameterProxy(weight)")

    def test_custom_parameter_proxy_shape(self):
        param = torch.nn.Parameter(torch.randn(3, 4))
        proxy = self._make_proxy(param)
        self.assertEqual(proxy.shape, param.shape)
        self.assertEqual(proxy.shape, torch.Size([3, 4]))

    def test_custom_parameter_proxy_size(self):
        param = torch.nn.Parameter(torch.randn(3, 4))
        proxy = self._make_proxy(param)
        self.assertEqual(proxy.size(), param.size())

    def test_custom_parameter_proxy_dim(self):
        param = torch.nn.Parameter(torch.randn(3, 4))
        proxy = self._make_proxy(param)
        self.assertEqual(proxy.dim(), 2)
        self.assertEqual(proxy.dim(), param.dim())

    def test_custom_parameter_proxy_ndim(self):
        param = torch.nn.Parameter(torch.randn(3, 4))
        proxy = self._make_proxy(param)
        self.assertEqual(proxy.ndim, 2)
        self.assertEqual(proxy.ndim, param.ndim)

    def test_custom_parameter_proxy_numel(self):
        param = torch.nn.Parameter(torch.randn(3, 4))
        proxy = self._make_proxy(param)
        self.assertEqual(proxy.numel(), 12)
        self.assertEqual(proxy.numel(), param.numel())

    def test_custom_parameter_proxy_nelement(self):
        param = torch.nn.Parameter(torch.randn(3, 4))
        proxy = self._make_proxy(param)
        self.assertEqual(proxy.nelement(), 12)
        self.assertEqual(proxy.nelement(), param.nelement())

    def test_custom_parameter_proxy_is_instance(self):
        param = torch.nn.Parameter(torch.randn(3, 4))
        proxy = self._make_proxy(param)
        self.assertIsInstance(proxy, CustomParameterProxy)
        self.assertIsInstance(proxy, CustomProxy)

    def test_custom_parameter_proxy_tracing_param_shapes_constant(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))

            def forward(self, x):
                # Access weight.shape during tracing; with param_shapes_constant=True
                # this should return actual shape values via CustomParameterProxy
                if self.weight.shape[0] == 4:
                    return x @ self.weight.T
                return x

        model = Model()
        tracer = CustomTracer(param_shapes_constant=True)
        graph = tracer.trace(model)
        self.assertIsNotNone(graph)
        node_ops = [n.op for n in graph.nodes]
        self.assertIn("get_attr", node_ops)

    def test_custom_parameter_proxy_1d(self):
        param = torch.nn.Parameter(torch.randn(5))
        proxy = self._make_proxy(param, name="bias")
        self.assertEqual(repr(proxy), "CustomParameterProxy(bias)")
        self.assertEqual(proxy.shape, torch.Size([5]))
        self.assertEqual(proxy.dim(), 1)
        self.assertEqual(proxy.ndim, 1)
        self.assertEqual(proxy.numel(), 5)
        self.assertEqual(proxy.nelement(), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
