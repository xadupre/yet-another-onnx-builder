import os
import tempfile
import unittest
import torch
from yobx.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings, requires_torch
from yobx.helpers.helper import get_sig_kwargs
from yobx.torch.export_options import (
    ExportOptions,
    TracingMode,
    _get_decomposition_table_by_name,
    _inplace_nodes,
    apply_decompositions,
    insert_contiguous_between_transpose_and_view,
)


class _Neuron(torch.nn.Module):
    """Simple Linear+relu model used as a shared test fixture."""

    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        return torch.relu(self.linear(x))


@requires_torch("2.0")
class TestExportOptions(ExtTestCase):
    def test_default_init(self):
        opts = ExportOptions()
        self.assertFalse(opts.strict)
        self.assertFalse(opts.jit)
        self.assertFalse(opts.dynamo)
        self.assertIsNone(opts.decomposition_table)
        self.assertIsNone(opts.strategy)
        self.assertTrue(opts.remove_inplace)
        self.assertFalse(opts.allow_untyped_output)
        self.assertIsNone(opts.save_ep)
        self.assertFalse(opts.validate_ep)
        self.assertEqual(opts.backed_size_oblivious, "auto")
        self.assertTrue(opts.prefer_deferred_runtime_asserts_over_guards)
        self.assertFalse(opts.fake)

    def test_repr_default(self):
        opts = ExportOptions()
        r = repr(opts)
        self.assertIsInstance(r, str)
        self.assertIn("ExportOptions", r)

    def test_repr_non_default(self):
        opts = ExportOptions(strict=True, decomposition_table="default")
        r = repr(opts)
        self.assertIn("strict=True", r)
        self.assertIn("decomposition_table='default'", r)

    def test_strategy_strict(self):
        opts = ExportOptions(strategy="strict")
        self.assertTrue(opts.strict)

    def test_strategy_nostrict(self):
        opts = ExportOptions(strategy="nostrict")
        self.assertFalse(opts.strict)

    def test_strategy_jit(self):
        opts = ExportOptions(strategy="jit")
        self.assertTrue(opts.jit)

    @ignore_warnings(FutureWarning)
    def test_strategy_dec(self):
        opts = ExportOptions(strategy="dec")
        self.assertEqual(opts.decomposition_table, "default")

    @ignore_warnings(FutureWarning)
    def test_strategy_decall(self):
        opts = ExportOptions(strategy="decall")
        self.assertEqual(opts.decomposition_table, "all")

    def test_strategy_fake(self):
        opts = ExportOptions(strategy="fake")
        self.assertTrue(opts.fake)

    def test_strategy_invalid(self):
        with self.assertRaises(AssertionError):
            ExportOptions(strategy="not-a-valid-strategy")

    def test_dynamo_jit_exclusive(self):
        with self.assertRaises(AssertionError):
            ExportOptions(dynamo=True, jit=True)

    def test_decomposition_table_none_for_none(self):
        opts = ExportOptions(decomposition_table=None)
        self.assertIsNone(opts.decomposition_table)

    def test_decomposition_table_none_for_none_string(self):
        opts = ExportOptions(decomposition_table="none")
        self.assertIsNone(opts.decomposition_table)

    def test_clone_no_changes(self):
        opts = ExportOptions(strict=True, decomposition_table="default")
        cloned = opts.clone()
        self.assertTrue(cloned.strict)
        self.assertEqual(cloned.decomposition_table, "default")

    def test_clone_with_override(self):
        opts = ExportOptions(strict=True, decomposition_table="default")
        cloned = opts.clone(strict=False)
        self.assertFalse(cloned.strict)
        self.assertEqual(cloned.decomposition_table, "default")

    def test_allowed_strategies(self):
        for strategy in ExportOptions._allowed:
            if strategy is None:
                continue
            opts = ExportOptions(strategy=strategy)
            self.assertIsNotNone(opts)

    def test_use_str_not_dyn_passthrough_int(self):
        opts = ExportOptions()
        self.assertEqual(opts.use_str_not_dyn(42), 42)

    def test_use_str_not_dyn_passthrough_str(self):
        opts = ExportOptions()
        self.assertEqual(opts.use_str_not_dyn("dim"), "dim")

    def test_use_str_not_dyn_none(self):
        opts = ExportOptions()
        self.assertIsNone(opts.use_str_not_dyn(None))

    def test_use_str_not_dyn_replaces_object(self):
        opts = ExportOptions()
        dim = torch.export.Dim("batch")
        result = opts.use_str_not_dyn(dim)
        self.assertIsInstance(result, str)

    def test_use_str_not_dyn_dict(self):
        opts = ExportOptions()
        dim = torch.export.Dim("batch")
        result = opts.use_str_not_dyn({"batch": dim, "seq": 5})
        self.assertIsInstance(result["batch"], str)
        self.assertEqual(result["seq"], 5)

    def test_use_str_not_dyn_list(self):
        opts = ExportOptions()
        dim = torch.export.Dim("batch")
        result = opts.use_str_not_dyn([dim, 3])
        self.assertIsInstance(result[0], str)
        self.assertEqual(result[1], 3)

    def test_export_simple_model(self):
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModel()
        opts = ExportOptions()
        ep = opts.export(
            model,
            args=(torch.randn(2, 3),),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @ignore_warnings(FutureWarning)
    def test_export_strict_true(self):
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x * 2

        model = SimpleModel()
        opts = ExportOptions(strict=True)
        ep = opts.export(
            model,
            args=(torch.randn(2, 3),),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    def test_export_neuron_model(self):
        """Neuron (Linear+relu) model export matches upstream Neuron fixture."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions()
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    def test_export_with_dynamic_shapes(self):
        """Export a Neuron model with a dynamic batch dimension."""
        model = _Neuron()
        x = torch.rand(2, 5)
        batch = torch.export.Dim("batch")
        opts = ExportOptions()
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=({0: batch},),
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @ignore_warnings(UserWarning)
    def test_export_with_jit(self):
        """Export with jit=True using TS2EPConverter."""
        try:
            from torch._export.converter import TS2EPConverter  # noqa: F401
        except ImportError:
            self.skipTest("TS2EPConverter not available in this torch version")

        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(jit=True)
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @ignore_warnings(FutureWarning)
    def test_export_with_decomposition_default(self):
        """Export with dec strategy applies default decompositions."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="dec")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @ignore_warnings(FutureWarning)
    def test_export_with_decomposition_all(self):
        """Export with decall strategy applies all decompositions."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="decall")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    def test_export_strategy_none(self):
        """Strategy 'none' behaves like the default export."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="none")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    def test_export_strategy_nostrict(self):
        """Strategy 'nostrict' exports with strict=False."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="nostrict")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @ignore_warnings(FutureWarning)
    def test_export_strategy_strict_dec(self):
        """Strategy 'strict-dec' exports with strict=True and default decompositions."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="strict-dec")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @ignore_warnings(FutureWarning)
    def test_export_strategy_strict_decall(self):
        """Strategy 'strict-decall' exports with strict=True and all decompositions."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="strict-decall")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @ignore_warnings(FutureWarning)
    def test_export_strategy_nostrict_dec(self):
        """Strategy 'nostrict-dec' exports with strict=False and default decompositions."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="nostrict-dec")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @ignore_warnings(FutureWarning)
    def test_export_strategy_nostrict_decall(self):
        """Strategy 'nostrict-decall' exports with strict=False and all decompositions."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="nostrict-decall")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @ignore_warnings((UserWarning, FutureWarning))
    def test_export_strategy_jit_dec(self):
        """Strategy 'jit-dec' exports via JIT with default decompositions."""
        try:
            from torch._export.converter import TS2EPConverter  # noqa: F401
        except ImportError:
            self.skipTest("TS2EPConverter not available in this torch version")
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="jit-dec")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @ignore_warnings((UserWarning, FutureWarning))
    def test_export_strategy_jit_decall(self):
        """Strategy 'jit-decall' exports via JIT with all decompositions."""
        try:
            from torch._export.converter import TS2EPConverter  # noqa: F401
        except ImportError:
            self.skipTest("TS2EPConverter not available in this torch version")
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="jit-decall")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    def test_export_strategy_fake(self):
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="fake")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_export_with_verbosity(self):
        """Export a simple model with verbose=1 to exercise the verbose code paths."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions()
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
            verbose=1,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)

    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_post_process_with_verbosity(self):
        """post_process_exported_program with verbose=1 exercises the verbose code paths."""
        model = _Neuron()
        x = torch.rand(2, 5)
        ep = torch.export.export(model, (x,))
        opts = ExportOptions(decomposition_table="default")
        result = opts.post_process_exported_program(ep, verbose=1)
        self.assertIsInstance(result, torch.export.ExportedProgram)

    def test_export_with_save_ep_true(self):
        """export() with save_ep=<str> saves the exported program files to disk."""
        model = _Neuron()
        x = torch.rand(2, 5)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            opts = ExportOptions(save_ep=save_path)
            ep = opts.export(
                model,
                args=(x,),
                kwargs=None,
                tracing_mode=False,
                dynamic_shapes=None,
                same_signature=True,
            )
            self.assertIsInstance(ep, torch.export.ExportedProgram)
            self.assertTrue(os.path.isfile(f"{save_path}.ep"))
            self.assertTrue(os.path.isfile(f"{save_path}.ep.graph"))
            self.assertTrue(os.path.isfile(f"{save_path}.input.pt"))
            self.assertTrue(os.path.isfile(f"{save_path}.ep.pt2"))

    def test_export_with_save_ep_tuple(self):
        """export() with save_ep=(str, int) respects the size threshold."""
        model = _Neuron()
        x = torch.rand(2, 5)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            # threshold of 0 means the model is always too large to save .pt files
            opts = ExportOptions(save_ep=(save_path, 0))
            ep = opts.export(
                model,
                args=(x,),
                kwargs=None,
                tracing_mode=False,
                dynamic_shapes=None,
                same_signature=True,
            )
            self.assertIsInstance(ep, torch.export.ExportedProgram)
            self.assertTrue(os.path.isfile(f"{save_path}.ep"))
            self.assertTrue(os.path.isfile(f"{save_path}.ep.graph"))
            # With threshold=0, model is too big so .pt files should NOT be saved
            self.assertFalse(os.path.isfile(f"{save_path}.input.pt"))
            self.assertFalse(os.path.isfile(f"{save_path}.ep.pt2"))

    def test_tracing_mode_new_tracing_enum_value(self):
        """Verifies that TracingMode.NEW_TRACING has the expected string value."""
        self.assertEqual(TracingMode.NEW_TRACING, "new-tracing")

    def test_tracing_mode_new_tracing_init(self):
        """Verifies that ExportOptions(tracing=TracingMode.NEW_TRACING) initializes correctly."""
        opts = ExportOptions(tracing=TracingMode.NEW_TRACING)
        self.assertEqual(opts.tracing, TracingMode.NEW_TRACING)

    def test_tracing_mode_new_tracing_string_init(self):
        """Verifies that ExportOptions(tracing='new-tracing') normalizes to NEW_TRACING."""
        opts = ExportOptions(tracing="new-tracing")
        self.assertEqual(opts.tracing, TracingMode.NEW_TRACING)

    def test_strategy_new_tracing(self):
        """Verifies that ExportOptions(strategy='new-tracing') sets tracing to NEW_TRACING."""
        opts = ExportOptions(strategy="new-tracing")
        self.assertEqual(opts.tracing, TracingMode.NEW_TRACING)

    def test_tracing_mode_new_tracing_incompatible_with_dynamo(self):
        """Verifies that NEW_TRACING and dynamo=True are rejected."""
        with self.assertRaises(AssertionError):
            ExportOptions(tracing=TracingMode.NEW_TRACING, dynamo=True)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_returns_graph_module(self):
        """Verifies that export() with NEW_TRACING returns a torch.fx.GraphModule."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(tracing=TracingMode.NEW_TRACING)
        gm = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(gm, torch.fx.GraphModule)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_param_placeholders_have_actual_weights(self):
        """Verifies that after NEW_TRACING export, parameter placeholder nodes have
        actual weights in meta['val']."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(tracing=TracingMode.NEW_TRACING)
        gm = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        # Parameter placeholder nodes must retain their actual weight tensor in meta["val"].
        param_names = {name for name, _ in model.named_parameters()}
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.meta.get("torch_name") in param_names:
                val = node.meta.get("val")
                self.assertIsInstance(
                    val,
                    torch.Tensor,
                    f"Parameter placeholder {node.name!r} should have an actual tensor "
                    "in meta['val']",
                )
                # Must NOT be a TracingTensor subclass.
                self.assertNotIn(
                    "TracingTensor",
                    type(val).__name__,
                    f"Parameter placeholder {node.name!r} meta['val'] must not be "
                    "a TracingTensor",
                )

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_to_onnx(self):
        """Verifies that to_onnx with ExportOptions(tracing=TracingMode.NEW_TRACING) succeeds."""
        import onnx
        from yobx.torch.interpreter import to_onnx

        model = _Neuron()
        x = torch.rand(2, 5)
        artifact = to_onnx(
            model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        onx = artifact.model_proto
        self.assertIsInstance(onx, onnx.ModelProto)
        # The ONNX model must have exactly one graph input ("x").
        self.assertEqual(len(onx.graph.input), 1)
        self.assertEqual(len(onx.graph.output), 1)


@requires_torch("2.0")
class TestApplyDecompositions(ExtTestCase):
    def test_apply_decompositions_none(self):
        ep = torch.export.export(_Neuron(), (torch.rand(2, 5),))
        result = apply_decompositions(ep, None, False)
        self.assertIs(result, ep)

    @ignore_warnings(FutureWarning)
    def test_apply_decompositions_default(self):
        ep = torch.export.export(_Neuron(), (torch.rand(2, 5),))
        result = apply_decompositions(ep, "default", False)
        self.assertIsInstance(result, torch.export.ExportedProgram)

    @ignore_warnings(FutureWarning)
    def test_apply_decompositions_all(self):
        ep = torch.export.export(_Neuron(), (torch.rand(2, 5),))
        result = apply_decompositions(ep, "all", False)
        self.assertIsInstance(result, torch.export.ExportedProgram)


@requires_torch("2.0")
class TestPostProcessExportedProgram(ExtTestCase):
    def test_post_process_no_decomposition(self):
        """post_process_exported_program with no decomposition table returns same program."""
        ep = torch.export.export(_Neuron(), (torch.rand(2, 5),))
        opts = ExportOptions()
        result = opts.post_process_exported_program(ep)
        self.assertIsInstance(result, torch.export.ExportedProgram)

    @ignore_warnings(FutureWarning)
    def test_post_process_with_decomposition(self):
        """post_process_exported_program with decomposition applies decompositions."""
        ep = torch.export.export(_Neuron(), (torch.rand(2, 5),))
        opts = ExportOptions(decomposition_table="default")
        result = opts.post_process_exported_program(ep)
        self.assertIsInstance(result, torch.export.ExportedProgram)


@requires_torch("2.0")
class TestInsertContiguous(ExtTestCase):
    def test_insert_contiguous_no_op(self):
        """With no transpose->view pattern, the program is returned unchanged."""

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        ep = torch.export.export(SimpleModel(), (torch.randn(2, 3),))
        result = insert_contiguous_between_transpose_and_view(ep)
        self.assertIs(result, ep)

    def test_insert_contiguous_call_method_transpose_view(self):
        """A contiguous node is inserted between call_method transpose and view nodes."""

        # Build a raw fx.Graph with transpose -> view call_method nodes
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        t = graph.call_method("transpose", args=(x, 0, 1))
        v = graph.call_method("view", args=(t, -1))
        graph.output(v)
        gm = torch.fx.GraphModule({}, graph)

        class _FakeEP:
            def __init__(self, gm):
                self.graph_module = gm

        fake_ep = _FakeEP(gm)
        result = insert_contiguous_between_transpose_and_view(fake_ep)

        contiguous_nodes = [
            n
            for n in result.graph_module.graph.nodes
            if n.op == "call_method" and n.target == "contiguous"
        ]
        self.assertEqual(len(contiguous_nodes), 1)
        # Verify graph structure: contiguous is between transpose and view
        contiguous_node = contiguous_nodes[0]
        self.assertEqual(contiguous_node.args[0].target, "transpose")
        view_users = list(contiguous_node.users)
        self.assertEqual(len(view_users), 1)
        self.assertEqual(view_users[0].target, "view")

    def test_insert_contiguous_aten_transpose_view(self):
        """A contiguous node is inserted between aten::transpose.int and aten::view."""

        class TransposeViewModel(torch.nn.Module):
            def forward(self, x):
                return x.transpose(0, 1).transpose(0, 1).view(-1)

        TransposeViewModel()(torch.randn(2, 3))
        ep = torch.export.export(TransposeViewModel(), (torch.randn(2, 3),))

        # Only run the insertion check when the pattern is present in the graph;
        # some torch versions normalise .view() to aten::reshape instead.
        has_pattern = any(
            n.op == "call_function"
            and hasattr(n.target, "name")
            and n.target.name() == "aten::transpose.int"
            and any(
                u.op == "call_function"
                and hasattr(u.target, "name")
                and u.target.name() == "aten::view"
                for u in n.users
            )
            for n in ep.graph_module.graph.nodes
        )

        result = insert_contiguous_between_transpose_and_view(ep)

        if has_pattern:
            contiguous_nodes = [
                n
                for n in result.graph_module.graph.nodes
                if n.op == "call_method" and n.target == "contiguous"
            ]
            self.assertGreater(len(contiguous_nodes), 0)
            # Verify graph structure: contiguous sits between transpose and view
            contiguous_node = contiguous_nodes[0]
            self.assertEqual(contiguous_node.args[0].target.name(), "aten::transpose.int")
            view_users = [
                u
                for u in contiguous_node.users
                if u.op == "call_function"
                and hasattr(u.target, "name")
                and u.target.name() == "aten::view"
            ]
            self.assertGreater(len(view_users), 0)
        else:
            # Pattern absent: function is a no-op and returns the same object
            self.assertIs(result, ep)


@requires_torch("2.0")
class TestInplaceFunctions(ExtTestCase):
    def test_inplace_nodes_empty_graph(self):
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        ep = torch.export.export(SimpleModel(), (torch.randn(2, 3),))
        nodes = _inplace_nodes(ep.graph)
        self.assertIsInstance(nodes, list)


@requires_torch("2.0")
class TestGetDecompositionTable(ExtTestCase):
    def test_get_decomposition_table_none(self):
        """get_decomposition_table returns None when decomposition_table is None."""
        opts = ExportOptions(decomposition_table=None)
        self.assertIsNone(opts.get_decomposition_table())

    def test_get_decomposition_table_all(self):
        """get_decomposition_table returns None when decomposition_table is 'all'."""
        opts = ExportOptions(decomposition_table="all")
        self.assertIsNone(opts.get_decomposition_table())

    def test_get_decomposition_table_default(self):
        """get_decomposition_table returns a dict when decomposition_table is 'default'."""
        opts = ExportOptions(decomposition_table="default")
        table = opts.get_decomposition_table()
        self.assertIsInstance(table, dict)
        self.assertGreater(len(table), 0)

    def test_get_decomposition_table_dict(self):
        """get_decomposition_table returns the dict directly when given a dict."""
        custom = {object(): lambda: None}
        opts = ExportOptions(decomposition_table=custom)
        self.assertIs(opts.get_decomposition_table(), custom)

    def test_get_decomposition_table_by_name_default(self):
        """_get_decomposition_table_by_name('default') returns a non-empty dict."""
        table = _get_decomposition_table_by_name("default")
        self.assertIsInstance(table, dict)
        self.assertGreater(len(table), 0)

    def test_get_decomposition_table_by_name_unknown(self):
        """_get_decomposition_table_by_name raises ValueError for unknown names."""
        with self.assertRaises(ValueError):
            _get_decomposition_table_by_name("unknown")


class TestValidateExportedProgram(ExtTestCase):
    def test_validate_exported_program_no_discrepancy(self):
        """validate_exported_program passes when model and exported program agree."""
        model = _Neuron()
        x = torch.rand(2, 5)
        ep = torch.export.export(model, (x,))
        opts = ExportOptions(validate_ep=True)
        # Should not raise
        opts.validate_exported_program(model, ep, (x,), None)

    def test_validate_exported_program_custom_atol(self):
        """validate_exported_program passes with a custom float atol."""
        model = _Neuron()
        x = torch.rand(2, 5)
        ep = torch.export.export(model, (x,))
        opts = ExportOptions(validate_ep=1e-3)
        opts.validate_exported_program(model, ep, (x,), None)

    def test_validate_exported_program_raises_on_discrepancy(self):
        """validate_exported_program raises AssertionError when outputs differ."""

        class ConstantModel(torch.nn.Module):
            def forward(self, x):
                return x + 1.0

        class MismatchedModel(torch.nn.Module):
            def forward(self, x):
                return x + 100.0

        x = torch.rand(2, 3)
        ep = torch.export.export(MismatchedModel(), (x,))
        reference_model = ConstantModel()
        opts = ExportOptions(validate_ep=True)
        with self.assertRaises(AssertionError):
            opts.validate_exported_program(reference_model, ep, (x,), None)

    def test_export_with_validate_ep_true(self):
        """export() with validate_ep=True runs validate_exported_program internally."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(validate_ep=True)
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)
        self.assertTrue(hasattr(opts, "_stat_time_validate_exported_program"))

    def test_export_with_validate_ep_float(self):
        """export() with validate_ep as float uses it as atol."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(validate_ep=1e-4)
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)
        self.assertTrue(hasattr(opts, "_stat_time_validate_exported_program"))


@requires_torch("2.0")
class TestGetSigKwargs(ExtTestCase):
    def test_get_sig_kwargs_returns_correct_values(self):
        opts = ExportOptions(strict=True, decomposition_table="default")
        kw = get_sig_kwargs(opts)
        self.assertEqual(kw["strict"], True)
        self.assertEqual(kw["decomposition_table"], "default")
        self.assertFalse(kw["jit"])


class TestRemoveInline(ExtTestCase):
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_remove_inline_slice_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                K_33[2:-2, 2:-2, :-1] = sumx[None, :, None]
                K_33[2:-2, 2:-2, -1] = 0.0
                return K_33

        model = Model()
        xs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )

        from yobx.torch.export_options import ExportOptions as _ExportOptions

        _opts = _ExportOptions()
        _ep = _opts.export(
            model,
            args=xs,
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=False,
            verbose=10,
        )
        for node in _ep.graph.nodes:
            if node.op == "output":
                continue
            self.assertGreater(
                len(node.users),
                0,
                msg=lambda: f"node with no users {node.name!r}\n{str(_ep.graph)}",
            )

        opts = ExportOptions()
        ep = opts.export(
            model,
            args=xs,
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=False,
            verbose=10,
        )
        for node in ep.graph.nodes:
            if node.op == "output":
                continue
            self.assertGreater(
                len(node.users),
                0,
                msg=lambda: f"node with no users {node.name!r}\n{str(ep.graph)}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
