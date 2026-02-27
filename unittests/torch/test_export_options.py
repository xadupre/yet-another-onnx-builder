"""Tests for ExportOptions in yobx.torch.export_options."""
import unittest
import torch
from yobx.ext_test_case import ExtTestCase, ignore_warnings, requires_torch
from yobx.helpers.helper import get_sig_kwargs
from yobx.torch.export_options import (
    ExportOptions,
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
        self.assertFalse(opts.fallback)
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

    def test_strategy_fallback(self):
        opts = ExportOptions(strategy="fallback")
        self.assertTrue(opts.fallback)

    def test_strategy_dec(self):
        opts = ExportOptions(strategy="dec")
        self.assertEqual(opts.decomposition_table, "default")

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

    def test_get_fallback_options_default(self):
        opts = ExportOptions()
        fallback = opts.get_fallback_options()
        self.assertIsInstance(fallback, list)
        self.assertGreater(len(fallback), 0)
        for f in fallback:
            self.assertIsInstance(f, ExportOptions)

    def test_get_fallback_options_strict(self):
        opts = ExportOptions()
        fallback = opts.get_fallback_options("strict")
        self.assertEqual(len(fallback), 2)

    def test_get_fallback_options_nostrict(self):
        opts = ExportOptions()
        fallback = opts.get_fallback_options("nostrict")
        self.assertEqual(len(fallback), 2)

    def test_get_fallback_options_jit(self):
        opts = ExportOptions()
        fallback = opts.get_fallback_options("jit")
        self.assertEqual(len(fallback), 2)

    def test_get_fallback_options_invalid_kind(self):
        opts = ExportOptions()
        with self.assertRaises(AssertionError):
            opts.get_fallback_options("not-a-kind")

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

    def test_export_fallback_strategy(self):
        """Fallback strategy tries multiple options and returns the first successful one."""
        model = _Neuron()
        x = torch.rand(2, 5)
        opts = ExportOptions(strategy="fallback")
        ep = opts.export(
            model,
            args=(x,),
            kwargs=None,
            tracing_mode=False,
            dynamic_shapes=None,
            same_signature=True,
        )
        self.assertIsInstance(ep, torch.export.ExportedProgram)
        # The winning fallback option is recorded
        self.assertIsNotNone(opts._last_working)


@requires_torch("2.0")
class TestApplyDecompositions(ExtTestCase):
    def test_apply_decompositions_none(self):
        """apply_decompositions with None table is a no-op and returns the same object."""
        ep = torch.export.export(_Neuron(), (torch.rand(2, 5),))
        result = apply_decompositions(ep, None, False)
        self.assertIs(result, ep)

    def test_apply_decompositions_default(self):
        """apply_decompositions with 'default' table produces an ExportedProgram."""
        ep = torch.export.export(_Neuron(), (torch.rand(2, 5),))
        result = apply_decompositions(ep, "default", False)
        self.assertIsInstance(result, torch.export.ExportedProgram)

    def test_apply_decompositions_all(self):
        """apply_decompositions with 'all' runs full decompositions."""
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
class TestGetSigKwargs(ExtTestCase):
    def test_get_sig_kwargs_returns_correct_values(self):
        opts = ExportOptions(strict=True, decomposition_table="default")
        kw = get_sig_kwargs(opts)
        self.assertEqual(kw["strict"], True)
        self.assertEqual(kw["decomposition_table"], "default")
        self.assertFalse(kw["fallback"])
        self.assertFalse(kw["jit"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
