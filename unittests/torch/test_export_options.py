"""Tests for ExportOptions in yobx.torch.export_options."""
import unittest
import torch
from yobx.ext_test_case import ExtTestCase, requires_torch
from yobx.helpers.helper import get_sig_kwargs
from yobx.torch.export_options import (
    ExportOptions,
    _inplace_nodes,
    insert_contiguous_between_transpose_and_view,
)


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
