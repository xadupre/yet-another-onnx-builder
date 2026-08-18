import os
import tempfile
import unittest
from yobx._onnx_shim import onnx  # noqa: TID251
import torch
from yobx.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings, requires_torch
from yobx.helpers.helper import get_sig_kwargs
from yobx.torch.export_options import (
    ExportOptions,
    TracingMode,
    ConvertingLibrary,
    _get_decomposition_table_by_name,
    _inplace_nodes,
    apply_decompositions,
    insert_contiguous_between_transpose_and_view,
)
from yobx.reference import ExtendedReferenceEvaluator
from yobx.torch.interpreter import to_onnx


class _Neuron(torch.nn.Module):
    """Simple Linear+relu model used as a shared test fixture."""

    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        return torch.relu(self.linear(x))


class _LinearModel(torch.nn.Module):
    """Dummy model with a single Linear layer, used to test TracingMode combinations."""

    def __init__(self, n_dims: int = 4, n_targets: int = 2):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        return self.linear(x)


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
        actual weights in meta['torch_value'] and retain their TracingTensor in meta['val']."""
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
        # Parameter placeholder nodes must carry the actual weight in meta["torch_value"]
        # while meta["val"] remains a TracingTensor (never overwritten).
        param_names = {name for name, _ in model.named_parameters()}
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.meta.get("torch_name") in param_names:
                torch_value = node.meta.get("torch_value")
                self.assertIsInstance(
                    torch_value,
                    torch.Tensor,
                    f"Parameter placeholder {node.name!r} should have an actual tensor "
                    "in meta['torch_value']",
                )
                # meta["val"] must remain as a TracingTensor (not overwritten).
                val = node.meta.get("val")
                self.assertIn(
                    "TracingTensor",
                    type(val).__name__,
                    f"Parameter placeholder {node.name!r} meta['val'] should be a "
                    "TracingTensor (not overwritten by export)",
                )

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_to_onnx(self):
        """Verifies that to_onnx with ExportOptions(tracing=TracingMode.NEW_TRACING) succeeds."""
        from yobx._onnx_shim import onnx  # noqa: TID251
        from yobx.torch.interpreter import to_onnx

        model = _Neuron()
        x = torch.rand(2, 5)
        artifact = to_onnx(
            model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        onx = artifact.proto
        self.assertIsInstance(onx, onnx.ModelProto)
        # The ONNX model must have exactly one graph input ("x").
        self.assertEqual(len(onx.graph.input), 1)
        self.assertEqual(len(onx.graph.output), 1)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_nanmean_to_onnx(self):
        """Checks that new tracing correctly exports nanmean operations."""

        class NanMeanModel(torch.nn.Module):
            def forward(self, x):
                return x.nanmean(dim=1, keepdim=True)

        model = NanMeanModel()
        x = torch.tensor(
            [[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]], dtype=torch.float32
        )
        expected = model(x)
        artifact = to_onnx(
            model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        self.assert_conversion_with_ort_on_cpu(artifact.proto, expected, (x,), atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_interpolate_antialias_to_onnx(self):
        """Checks that new tracing exports torch.nn.functional.interpolate
        with antialias=True (aten._upsample_bilinear2d_aa and
        aten._upsample_bicubic2d_aa). This pattern arises when exporting
        torchvision v2 pipelines that use ``v2.Resize(..., antialias=True)``.
        """

        class BilinearAA(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(
                    x, size=(8, 8), mode="bilinear", antialias=True
                )

        class BicubicAA(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(
                    x, size=(8, 8), mode="bicubic", antialias=True
                )

        x = torch.randn(1, 2, 16, 16, dtype=torch.float32)
        for model_cls in (BilinearAA, BicubicAA):
            with self.subTest(model=model_cls.__name__):
                model = model_cls()
                expected = model(x)
                artifact = to_onnx(
                    model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
                )
                # PyTorch and ONNX Resize implement compatible but not
                # bit-identical antialiased downsampling filters.
                self.assert_conversion_with_ort_on_cpu(artifact.proto, expected, (x,), atol=0.5)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_addcmul_inplace_to_onnx(self):
        """Checks that new tracing correctly exports inplace addcmul."""

        class AddcmulInplaceModel(torch.nn.Module):
            def forward(self, x, y, z):
                xc = x.clone()
                xc.addcmul_(y, z, value=0.25)
                return xc

        model = AddcmulInplaceModel()
        x = torch.rand(2, 3, dtype=torch.float32)
        y = torch.rand(2, 3, dtype=torch.float32)
        z = torch.rand(2, 3, dtype=torch.float32)
        expected = model(x, y, z)
        artifact = to_onnx(
            model, (x, y, z), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        self.assert_conversion_with_ort_on_cpu(artifact.proto, expected, (x, y, z), atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_isclose_isfinite_to_onnx(self):
        """Checks that new tracing correctly exports isclose and isfinite."""

        class IsCloseIsFiniteModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.isclose(x, y), torch.isfinite(x)

        model = IsCloseIsFiniteModel()
        x = torch.tensor([[1.0, float("nan")], [float("inf"), 3.0]], dtype=torch.float32)
        y = torch.tensor([[1.0, float("nan")], [0.0, 2.0]], dtype=torch.float32)
        expected = model(x, y)
        artifact = to_onnx(
            model, (x, y), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        self.assert_conversion_with_ort_on_cpu(artifact.proto, expected, (x, y), atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_bincount_to_onnx(self):
        """Checks that new tracing correctly exports bincount."""

        class BinCountModel(torch.nn.Module):
            def forward(self, x):
                return torch.bincount(x, minlength=8)

        model = BinCountModel()
        x = torch.tensor([0, 1, 1, 3, 2, 1], dtype=torch.int64)
        expected = model(x)
        artifact = to_onnx(
            model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        self.assert_conversion_with_ort_on_cpu(artifact.proto, expected, (x,), atol=0)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_argsort_to_onnx(self):
        """Verifies that new tracing correctly exports argsort."""

        class ArgSortModel(torch.nn.Module):
            def forward(self, x):
                return torch.argsort(x, dim=-1, descending=True), x.argsort(dim=0)

        model = ArgSortModel()
        x = torch.tensor([[0.2, 4.0, -1.0], [7.0, -3.0, 2.0]], dtype=torch.float32)

        expected = model(x)
        artifact = to_onnx(
            model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        self.assert_conversion_with_ort_on_cpu(artifact.proto, expected, (x,), atol=0)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_cast_double_to_onnx(self):
        """Checks that new tracing correctly exports .double() (cast to float64)."""

        class CastDoubleModel(torch.nn.Module):
            def forward(self, x):
                return x.double()

        model = CastDoubleModel()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        expected = model(x)
        artifact = to_onnx(
            model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        self.assert_conversion_with_ort_on_cpu(artifact.proto, expected, (x,), atol=1e-6)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_cast_float_to_onnx(self):
        """Checks that new tracing correctly exports .float() (cast to float32)."""

        class CastFloatModel(torch.nn.Module):
            def forward(self, x):
                return x.float()

        model = CastFloatModel()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        expected = model(x)
        artifact = to_onnx(
            model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        self.assert_conversion_with_ort_on_cpu(artifact.proto, expected, (x,), atol=1e-6)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_cast_byte_to_onnx(self):
        """Checks that new tracing correctly exports .byte() (cast to uint8)."""

        class CastByteModel(torch.nn.Module):
            def forward(self, x):
                return x.byte()

        model = CastByteModel()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        expected = model(x)
        artifact = to_onnx(
            model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        self.assert_conversion_with_ort_on_cpu(artifact.proto, expected, (x,), atol=0)

    @ignore_warnings(UserWarning)
    def test_export_new_tracing_cast_bool_to_onnx(self):
        """Checks that new tracing correctly exports .bool() (cast to bool)."""

        class CastBoolModel(torch.nn.Module):
            def forward(self, x):
                return x.bool()

        model = CastBoolModel()
        x = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
        expected = model(x)
        artifact = to_onnx(
            model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        self.assert_conversion_with_ort_on_cpu(artifact.proto, expected, (x,), atol=0)


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

    def _scaled_mm_inputs(self, device: str = "cpu"):
        x = torch.randn((2, 3), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
        y = torch.randn((3, 4), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
        return (
            x,
            y,
            torch.tensor(0.5, dtype=torch.float32, device=device),
            torch.tensor(2.0, dtype=torch.float32, device=device),
        )

    def _assert_reference_conversion(self, onx, expected, inputs, atol=0, rtol=0):
        sess = ExtendedReferenceEvaluator(onx)
        feeds = dict(zip(sess.input_names, [self.to_numpy(x) for x in inputs]))
        got = sess.run(None, feeds)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0], atol=atol, rtol=rtol)

    def _scaled_mm_reference(self, inputs, out_dtype=torch.float16):
        x, y, scale_a, scale_b = inputs
        return torch.matmul(
            x.to(torch.float32) * scale_a.to(torch.float32),
            y.to(torch.float32) * scale_b.to(torch.float32),
        ).to(out_dtype)

    def _check_scaled_mm_model(self, model, inputs=None, expected=None):
        inputs = self._scaled_mm_inputs() if inputs is None else inputs
        expected = model(*inputs) if expected is None else expected
        for tracing in (None, TracingMode.TRACING, TracingMode.NEW_TRACING):
            with self.subTest(tracing=tracing):
                export_options = (
                    ExportOptions() if tracing is None else ExportOptions(tracing=tracing)
                )
                artifact = to_onnx(model, inputs, export_options=export_options)
                self._assert_reference_conversion(artifact.proto, expected, inputs, atol=1e-2)

    @requires_torch("2.11")
    @ignore_warnings(UserWarning)
    def test_export_scaled_mm_to_onnx(self):
        """Checks that torch._scaled_mm exports in all tracing modes."""

        class ScaledMMModel(torch.nn.Module):
            def forward(self, x, y, scale_a, scale_b):
                return torch._scaled_mm(
                    x, y, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16
                )

        self._check_scaled_mm_model(ScaledMMModel())

    @requires_torch("2.11")
    @ignore_warnings(UserWarning)
    @unittest.skipIf(
        not torch.cuda.is_available(), "torch._scaled_mm_v2 is not implemented on CPU"
    )
    def test_export_scaled_mm_v2_to_onnx(self):
        """Checks that torch._scaled_mm_v2 exports in all tracing modes."""

        class ScaledMMV2Model(torch.nn.Module):
            def forward(self, x, y, scale_a, scale_b):
                return torch._scaled_mm_v2(
                    x,
                    y,
                    [scale_a],
                    [int(torch.nn.functional.ScalingType.TensorWise)],
                    [],
                    [scale_b],
                    [int(torch.nn.functional.ScalingType.TensorWise)],
                    [],
                    None,
                    torch.float16,
                    (),
                    False,
                )

        self._check_scaled_mm_model(ScaledMMV2Model(), inputs=self._scaled_mm_inputs("cuda"))

    @requires_torch("2.11")
    @ignore_warnings(UserWarning)
    def test_export_scaled_mm_v2_to_onnx_cpu_alternative(self):
        """Checks that scaled_mm_v2 lowering still works on CPU without executing the op."""

        from yobx.torch.interpreter._aten_functions import aten__scaled_mm_v2
        from yobx.torch.torch_helper import torch_dtype_to_onnx_dtype
        from yobx.xbuilder import GraphBuilder

        inputs = self._scaled_mm_inputs()
        x, y, scale_a, scale_b = inputs
        g = GraphBuilder({"": 19}, verbose=0)
        g.make_tensor_input("x", torch_dtype_to_onnx_dtype(x.dtype), x.shape)
        g.make_tensor_input("y", torch_dtype_to_onnx_dtype(y.dtype), y.shape)
        g.make_tensor_input("scale_a", torch_dtype_to_onnx_dtype(scale_a.dtype), scale_a.shape)
        g.make_tensor_input("scale_b", torch_dtype_to_onnx_dtype(scale_b.dtype), scale_b.shape)
        tensorwise = int(torch.nn.functional.ScalingType.TensorWise)
        output = aten__scaled_mm_v2(
            g,
            None,
            ["scaled_mm_v2"],
            "x",
            "y",
            ["scale_a"],
            [tensorwise],
            [],
            ["scale_b"],
            [tensorwise],
            [],
            None,
            torch.float16,
            (),
            False,
        )
        g.make_tensor_output(output, torch_dtype_to_onnx_dtype(torch.float16), (2, 4))
        self._assert_reference_conversion(
            g.to_onnx(optimize=False), self._scaled_mm_reference(inputs), inputs, atol=1e-2
        )

    @requires_torch("2.11")
    @ignore_warnings(UserWarning)
    def test_export_scaled_mm_v2_rejects_non_tensorwise_recipe(self):
        """Checks that scaled_mm_v2 rejects recipe values the lowering ignores."""

        class ScaledMMV2RowWiseModel(torch.nn.Module):
            def forward(self, x, y, scale_a, scale_b):
                return torch._scaled_mm_v2(
                    x,
                    y,
                    [scale_a],
                    [int(torch.nn.functional.ScalingType.RowWise)],
                    [],
                    [scale_b],
                    [int(torch.nn.functional.ScalingType.RowWise)],
                    [],
                    None,
                    torch.float16,
                    (),
                    False,
                )

        model = ScaledMMV2RowWiseModel()
        inputs = (
            torch.randn((2, 3), dtype=torch.float32).to(torch.float8_e4m3fn),
            torch.randn((3, 4), dtype=torch.float32).to(torch.float8_e4m3fn),
            torch.ones((2, 1), dtype=torch.float32),
            torch.ones((1, 4), dtype=torch.float32),
        )
        with self.assertRaises(AssertionError):
            to_onnx(model, inputs, export_options=ExportOptions())


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


class TestConvertingLibrary(ExtTestCase):
    def test_converting_library_default_enum_value(self):
        """Verifies that ConvertingLibrary.DEFAULT has the expected string value."""
        self.assertEqual(ConvertingLibrary.DEFAULT, "default")

    def test_converting_library_onnxscript_enum_value(self):
        """Verifies that ConvertingLibrary.ONNXSCRIPT has the expected string value."""
        self.assertEqual(ConvertingLibrary.ONNXSCRIPT, "onnxscript")

    def test_converting_library_default_init(self):
        """Verifies that ExportOptions defaults to ConvertingLibrary.DEFAULT."""
        opts = ExportOptions()
        self.assertEqual(opts.converting_library, ConvertingLibrary.DEFAULT)

    def test_converting_library_enum_init(self):
        """Verifies that ExportOptions accepts a ConvertingLibrary enum value."""
        opts = ExportOptions(converting_library=ConvertingLibrary.ONNXSCRIPT)
        self.assertEqual(opts.converting_library, ConvertingLibrary.ONNXSCRIPT)

    def test_converting_library_string_init(self):
        """Verifies that ExportOptions accepts a string and normalizes to ConvertingLibrary."""
        opts = ExportOptions(converting_library="onnxscript")
        self.assertEqual(opts.converting_library, ConvertingLibrary.ONNXSCRIPT)

    def test_converting_library_invalid_string_raises(self):
        """Verifies that an invalid converting_library string raises ValueError."""
        with self.assertRaises(ValueError):
            ExportOptions(converting_library="invalid_lib")

    def test_strategy_onnxscript_sets_tracing(self):
        """Verifies that strategy='onnxscript' sets tracing to TracingMode.ONNXSCRIPT."""
        opts = ExportOptions(strategy="onnxscript")
        self.assertEqual(opts.tracing, TracingMode.ONNXSCRIPT)

    def test_strategy_onnxscript_implies_converting_library(self):
        """Verifies that strategy='onnxscript' also sets converting_library to ONNXSCRIPT."""
        opts = ExportOptions(strategy="onnxscript")
        self.assertEqual(opts.converting_library, ConvertingLibrary.ONNXSCRIPT)

    def test_converting_library_repr(self):
        """Verifies that converting_library appears in repr when non-default."""
        opts = ExportOptions(converting_library=ConvertingLibrary.ONNXSCRIPT)
        r = repr(opts)
        self.assertIn("ONNXSCRIPT", r)

    def test_converting_library_clone(self):
        """Verifies that clone() preserves converting_library."""
        opts = ExportOptions(converting_library=ConvertingLibrary.ONNXSCRIPT)
        cloned = opts.clone()
        self.assertEqual(cloned.converting_library, ConvertingLibrary.ONNXSCRIPT)

    def test_tracing_mode_onnxscript_enum_value(self):
        """Verifies that TracingMode.ONNXSCRIPT has the expected string value."""
        self.assertEqual(TracingMode.ONNXSCRIPT, "onnxscript")

    def test_tracing_mode_onnxscript_sets_converting_library(self):
        """Verifies that tracing=TracingMode.ONNXSCRIPT automatically sets converting_library."""
        opts = ExportOptions(tracing=TracingMode.ONNXSCRIPT)
        self.assertEqual(opts.tracing, TracingMode.ONNXSCRIPT)
        self.assertEqual(opts.converting_library, ConvertingLibrary.ONNXSCRIPT)

    def test_tracing_mode_onnxscript_string_init(self):
        """Verifies that tracing='onnxscript' normalizes to TracingMode.ONNXSCRIPT."""
        opts = ExportOptions(tracing="onnxscript")
        self.assertEqual(opts.tracing, TracingMode.ONNXSCRIPT)
        self.assertEqual(opts.converting_library, ConvertingLibrary.ONNXSCRIPT)

    def test_tracing_mode_onnxscript_clone(self):
        """Verifies that clone() preserves tracing=TracingMode.ONNXSCRIPT."""
        opts = ExportOptions(tracing=TracingMode.ONNXSCRIPT)
        cloned = opts.clone()
        self.assertEqual(cloned.tracing, TracingMode.ONNXSCRIPT)
        self.assertEqual(cloned.converting_library, ConvertingLibrary.ONNXSCRIPT)


@requires_torch("2.0")
class TestToOnnxConvertingLibrary(ExtTestCase):
    """Verifies that :func:`to_onnx` routes correctly when
    ``TracingMode.ONNXSCRIPT`` or ``ConvertingLibrary.ONNXSCRIPT`` is used."""

    # A minimal toy model used by every test in this class.
    class _Add(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    @ignore_warnings(FutureWarning)
    def test_to_onnx_tracing_onnxscript_calls_dynamo_export(self):
        """Verifies that to_onnx with tracing=TracingMode.ONNXSCRIPT
        calls torch.onnx.export with dynamo=True."""
        from yobx._onnx_shim import onnx  # noqa: TID251
        from unittest.mock import MagicMock, patch

        model = self._Add()
        x = torch.randn(2, 3)

        fake_out = MagicMock()
        fake_out.model_proto = onnx.ModelProto()

        with patch("torch.onnx.export", return_value=fake_out) as mock_export:
            to_onnx(model, (x,), export_options=ExportOptions(tracing=TracingMode.ONNXSCRIPT))

        mock_export.assert_called_once()
        call_args = mock_export.call_args
        # first positional argument must be the nn.Module, not an ExportedProgram
        self.assertIsInstance(call_args.args[0], torch.nn.Module)
        # dynamo=True must be present
        self.assertTrue(call_args.kwargs.get("dynamo", False))

    @ignore_warnings(FutureWarning)
    def test_to_onnx_tracing_onnxscript_does_not_call_export_options_export(self):
        """Verifies that to_onnx with tracing=TracingMode.ONNXSCRIPT
        does not call ExportOptions.export()."""
        from yobx._onnx_shim import onnx  # noqa: TID251
        from unittest.mock import MagicMock, patch

        model = self._Add()
        x = torch.randn(2, 3)

        fake_out = MagicMock()
        fake_out.model_proto = onnx.ModelProto()

        with (
            patch("torch.onnx.export", return_value=fake_out),
            patch.object(ExportOptions, "export") as mock_opts_export,
        ):
            to_onnx(model, (x,), export_options=ExportOptions(tracing=TracingMode.ONNXSCRIPT))

        mock_opts_export.assert_not_called()

    @ignore_warnings(FutureWarning)
    def test_to_onnx_converting_library_onnxscript_uses_exported_program(self):
        """Verifies that to_onnx with converting_library=ONNXSCRIPT + default tracing uses
        ExportOptions.export() to obtain an ExportedProgram, then calls
        torch.onnx.export on that program (not with dynamo=True)."""
        from yobx._onnx_shim import onnx  # noqa: TID251
        from unittest.mock import MagicMock, patch

        model = self._Add()
        x = torch.randn(2, 3)

        fake_ep = MagicMock()
        fake_onnx_out = MagicMock()
        fake_onnx_out.model_proto = onnx.ModelProto()

        with (
            patch.object(ExportOptions, "export", return_value=fake_ep) as mock_opts_export,
            patch("torch.onnx.export", return_value=fake_onnx_out) as mock_torch_export,
        ):
            to_onnx(
                model,
                (x,),
                export_options=ExportOptions(converting_library=ConvertingLibrary.ONNXSCRIPT),
            )

        # ExportOptions.export() must have been called to produce the ExportedProgram.
        mock_opts_export.assert_called_once()
        # torch.onnx.export() must have been called with the ExportedProgram.
        mock_torch_export.assert_called_once()
        call_args = mock_torch_export.call_args
        self.assertIs(call_args.args[0], fake_ep)
        # dynamo=True must NOT appear — the EP path doesn't need it.
        self.assertNotIn("dynamo", call_args.kwargs)

    @ignore_warnings(FutureWarning)
    def test_to_onnx_strategy_onnxscript_calls_dynamo_export(self):
        """Verifies that to_onnx with strategy='onnxscript' (shorthand for TracingMode.ONNXSCRIPT)
        calls torch.onnx.export with dynamo=True on the original model."""
        from yobx._onnx_shim import onnx  # noqa: TID251
        from unittest.mock import MagicMock, patch

        model = self._Add()
        x = torch.randn(2, 3)

        fake_out = MagicMock()
        fake_out.model_proto = onnx.ModelProto()

        with patch("torch.onnx.export", return_value=fake_out) as mock_export:
            to_onnx(model, (x,), export_options=ExportOptions(strategy="onnxscript"))

        mock_export.assert_called_once()
        call_args = mock_export.call_args
        self.assertIsInstance(call_args.args[0], torch.nn.Module)
        self.assertTrue(call_args.kwargs.get("dynamo", False))


@requires_torch("2.0")
class TestTracingModeCombinationsLinear(ExtTestCase):
    """Tests every combination of TracingMode × ConvertingLibrary for a single Linear layer.

    All combinations are exercised end-to-end via
    :func:`~yobx.torch.interpreter.to_onnx` without any mocking.
    """

    # ------------------------------------------------------------------ helpers

    def _make_model(self) -> _LinearModel:
        return _LinearModel()

    def _make_input(self) -> torch.Tensor:
        return torch.randn(3, 4)

    def _assert_bitwise_not_export(
        self, x: torch.Tensor, tracing_mode: TracingMode, expected_op_type: str
    ) -> None:
        class Model(torch.nn.Module):
            def forward(self, y):
                return ~y

        model = Model()
        expected = model(x).detach().numpy()
        artifact = to_onnx(model, (x,), export_options=ExportOptions(tracing=tracing_mode))
        onx = artifact.proto
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertEqual([node.op_type for node in onx.graph.node], [expected_op_type])
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"y": x.numpy()})
        self.assertEqualArray(expected, got[0])

    def _assert_istft_export(self, tracing_mode: TracingMode) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.window = torch.hann_window(8)

            def forward(self, y):
                return torch.istft(
                    y,
                    n_fft=8,
                    hop_length=4,
                    win_length=8,
                    window=self.window,
                    center=True,
                    normalized=False,
                    onesided=True,
                    length=16,
                    return_complex=False,
                )

        model = Model()
        x = torch.randn((2, 5, 5), dtype=torch.complex64)
        expected = model(x).detach().numpy()
        artifact = to_onnx(model, (x,), export_options=ExportOptions(tracing=tracing_mode))
        onx = artifact.proto
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertIn("Istft", [node.op_type for node in onx.graph.node])
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"y": x.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    # ------------------------------------------------------------------ ConvertingLibrary.DEFAULT

    @ignore_warnings(UserWarning)
    def test_linear_default_default(self):
        """TracingMode.DEFAULT + ConvertingLibrary.DEFAULT: torch.export + yobx pipeline."""
        model = self._make_model()
        x = self._make_input()
        expected = model(x).detach().numpy()
        artifact = to_onnx(model, (x,), export_options=ExportOptions(tracing=TracingMode.DEFAULT))
        onx = artifact.proto
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertEqual(len(onx.graph.input), 1)
        self.assertEqual(len(onx.graph.output), 1)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_linear_tracing_default(self):
        """TracingMode.TRACING + ConvertingLibrary.DEFAULT: CustomTracer + yobx pipeline."""
        model = self._make_model()
        x = self._make_input()
        expected = model(x).detach().numpy()
        artifact = to_onnx(model, (x,), export_options=ExportOptions(tracing=TracingMode.TRACING))
        onx = artifact.proto
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertEqual(len(onx.graph.input), 1)
        self.assertEqual(len(onx.graph.output), 1)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_linear_new_tracing_default(self):
        """TracingMode.NEW_TRACING + ConvertingLibrary.DEFAULT: GraphTracer + yobx pipeline."""
        model = self._make_model()
        x = self._make_input()
        expected = model(x).detach().numpy()
        artifact = to_onnx(
            model, (x,), export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        onx = artifact.proto
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertEqual(len(onx.graph.input), 1)
        self.assertEqual(len(onx.graph.output), 1)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_bitwise_not_all_default_exporters(self):
        """Verifies that unary invert exports through all tracing modes."""
        cases = (
            (torch.tensor([[True, False], [False, True]]), "Not"),
            (torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), "BitwiseNot"),
        )
        for x, expected_op_type in cases:
            for tracing_mode in (
                TracingMode.DEFAULT,
                TracingMode.TRACING,
                TracingMode.NEW_TRACING,
            ):
                with self.subTest(dtype=x.dtype, tracing_mode=tracing_mode):
                    self._assert_bitwise_not_export(x, tracing_mode, expected_op_type)

    @ignore_warnings(UserWarning)
    def test_istft_all_default_exporters(self):
        """Verifies that istft exports through all tracing modes."""
        for tracing_mode in (TracingMode.DEFAULT, TracingMode.TRACING, TracingMode.NEW_TRACING):
            with self.subTest(tracing_mode=tracing_mode):
                self._assert_istft_export(tracing_mode)

    # --------------------------------------------------------------- ConvertingLibrary.ONNXSCRIPT

    @ignore_warnings((UserWarning, FutureWarning))
    def test_linear_default_onnxscript(self):
        """TracingMode.DEFAULT + ConvertingLibrary.ONNXSCRIPT: produces valid ONNX."""
        model = self._make_model()
        x = self._make_input()

        artifact = to_onnx(
            model,
            (x,),
            export_options=ExportOptions(
                tracing=TracingMode.DEFAULT, converting_library=ConvertingLibrary.ONNXSCRIPT
            ),
        )

        self.assertIsInstance(artifact.proto, onnx.ModelProto)

    @unittest.skip("onnxscript does not support fx.graph produced by TRACING mode")
    @ignore_warnings((UserWarning, FutureWarning))
    def test_linear_tracing_onnxscript(self):
        """TracingMode.TRACING + ConvertingLibrary.ONNXSCRIPT: produces valid ONNX."""
        model = self._make_model()
        x = self._make_input()

        artifact = to_onnx(
            model,
            (x,),
            export_options=ExportOptions(
                tracing=TracingMode.TRACING, converting_library=ConvertingLibrary.ONNXSCRIPT
            ),
        )

        self.assertIsInstance(artifact.proto, onnx.ModelProto)

    @unittest.skip("onnxscript does not support fx.graph produced by NEW_TRACING mode")
    @ignore_warnings((UserWarning, FutureWarning))
    def test_linear_new_tracing_onnxscript(self):
        """TracingMode.NEW_TRACING + ConvertingLibrary.ONNXSCRIPT: produces valid ONNX."""
        model = self._make_model()
        x = self._make_input()

        artifact = to_onnx(
            model,
            (x,),
            export_options=ExportOptions(
                tracing=TracingMode.NEW_TRACING, converting_library=ConvertingLibrary.ONNXSCRIPT
            ),
        )

        self.assertIsInstance(artifact.proto, onnx.ModelProto)

    @ignore_warnings(FutureWarning)
    def test_linear_onnxscript_onnxscript(self):
        """TracingMode.ONNXSCRIPT: torch.onnx.export called with dynamo=True."""
        from unittest.mock import MagicMock, patch

        model = self._make_model()
        x = self._make_input()

        fake_out = MagicMock()
        fake_out.model_proto = onnx.ModelProto()

        with patch("torch.onnx.export", return_value=fake_out) as mock_export:
            to_onnx(model, (x,), export_options=ExportOptions(tracing=TracingMode.ONNXSCRIPT))

        mock_export.assert_called_once()
        call_args = mock_export.call_args
        # The original nn.Module must be passed directly (not an ExportedProgram).
        self.assertIsInstance(call_args.args[0], torch.nn.Module)
        # dynamo=True is required for this path.
        self.assertTrue(call_args.kwargs.get("dynamo", False))


@requires_torch("2.0")
class TestTracingModeCombinationsBitwiseXor(ExtTestCase):
    """Tests bitwise XOR export across the supported tracing modes."""

    def _make_inputs(self):
        return (
            torch.tensor([1, 2, 7], dtype=torch.int64),
            torch.tensor([3, 1, 4], dtype=torch.int64),
        )

    def _assert_export(self, tracing_mode: TracingMode):
        class BitwiseXorModel(torch.nn.Module):
            def forward(self, x, y):
                return x.bitwise_xor(y)

        model = BitwiseXorModel()
        inputs = self._make_inputs()
        expected = model(*inputs).detach().numpy()
        artifact = to_onnx(model, inputs, export_options=ExportOptions(tracing=tracing_mode))
        ref = ExtendedReferenceEvaluator(artifact.proto)
        got = ref.run(None, {name: value.numpy() for name, value in zip(("x", "y"), inputs)})[0]
        self.assertEqualArray(expected, got)

    @ignore_warnings(UserWarning)
    def test_bitwise_xor_default_default(self):
        self._assert_export(TracingMode.DEFAULT)

    @ignore_warnings(UserWarning)
    def test_bitwise_xor_tracing_default(self):
        self._assert_export(TracingMode.TRACING)

    @ignore_warnings(UserWarning)
    def test_bitwise_xor_new_tracing_default(self):
        self._assert_export(TracingMode.NEW_TRACING)

    @ignore_warnings(UserWarning)
    def test_operator_xor_all_default_libraries(self):
        class BitwiseOperatorXorModel(torch.nn.Module):
            def forward(self, x, y):
                return x ^ y

        model = BitwiseOperatorXorModel()
        inputs = self._make_inputs()
        expected = model(*inputs).detach().numpy()
        for tracing_mode in (TracingMode.DEFAULT, TracingMode.TRACING, TracingMode.NEW_TRACING):
            with self.subTest(tracing_mode=tracing_mode):
                artifact = to_onnx(
                    model, inputs, export_options=ExportOptions(tracing=tracing_mode)
                )
                ref = ExtendedReferenceEvaluator(artifact.proto)
                got = ref.run(
                    None, {name: value.numpy() for name, value in zip(("x", "y"), inputs)}
                )[0]
                self.assertEqualArray(expected, got)


@requires_torch("2.0")
class TestTracingModeCombinationsBitwiseAndOr(ExtTestCase):
    """Tests bitwise AND/OR export across the supported tracing modes."""

    def _make_inputs(self):
        return (
            torch.tensor([1, 2, 7], dtype=torch.int64),
            torch.tensor([3, 1, 4], dtype=torch.int64),
        )

    def _assert_export(self, model: torch.nn.Module, tracing_mode: TracingMode):
        inputs = self._make_inputs()
        expected = tuple(output.detach().numpy() for output in model(*inputs))
        artifact = to_onnx(model, inputs, export_options=ExportOptions(tracing=tracing_mode))
        ref = ExtendedReferenceEvaluator(artifact.proto)
        got = ref.run(None, {name: value.numpy() for name, value in zip(("x", "y"), inputs)})
        self.assertEqual(len(expected), len(got))
        for exp, res in zip(expected, got):
            self.assertEqualArray(exp, res)

    @ignore_warnings(UserWarning)
    def test_bitwise_and_or_methods_all_default_libraries(self):
        class BitwiseMethodsModel(torch.nn.Module):
            def forward(self, x, y):
                return x.bitwise_and(y), x.bitwise_or(y), x.bitwise_and(1), x.bitwise_or(1)

        model = BitwiseMethodsModel()
        for tracing_mode in (TracingMode.DEFAULT, TracingMode.TRACING, TracingMode.NEW_TRACING):
            with self.subTest(tracing_mode=tracing_mode):
                self._assert_export(model, tracing_mode)

    @ignore_warnings(UserWarning)
    def test_operator_and_or_all_default_libraries(self):
        class BitwiseOperatorModel(torch.nn.Module):
            def forward(self, x, y):
                return x & y, x | y, x & 1, x | 1

        model = BitwiseOperatorModel()
        for tracing_mode in (TracingMode.DEFAULT, TracingMode.TRACING, TracingMode.NEW_TRACING):
            with self.subTest(tracing_mode=tracing_mode):
                self._assert_export(model, tracing_mode)


@requires_torch("2.0")
class TestTracingModeCombinationsBitwiseShifts(ExtTestCase):
    """Tests bitwise shift export across the supported tracing modes."""

    def _make_inputs(self):
        return (
            torch.tensor([1, 2, 7], dtype=torch.int64),
            torch.tensor([3, 1, 4], dtype=torch.int64),
        )

    def _assert_export(self, model: torch.nn.Module, tracing_mode: TracingMode):
        inputs = self._make_inputs()
        expected = tuple(output.detach().numpy() for output in model(*inputs))
        artifact = to_onnx(model, inputs, export_options=ExportOptions(tracing=tracing_mode))
        ref = ExtendedReferenceEvaluator(artifact.proto)
        got = ref.run(None, {name: value.numpy() for name, value in zip(("x", "y"), inputs)})
        self.assertEqual(len(expected), len(got))
        for exp, res in zip(expected, got):
            self.assertEqualArray(exp, res)

    @ignore_warnings(UserWarning)
    def test_bitwise_shift_methods_all_default_libraries(self):
        class BitwiseShiftMethodsModel(torch.nn.Module):
            def forward(self, x, y):
                return (
                    x.bitwise_left_shift(y),
                    x.bitwise_right_shift(y),
                    x.bitwise_left_shift(1),
                    x.bitwise_right_shift(1),
                )

        model = BitwiseShiftMethodsModel()
        for tracing_mode in (TracingMode.DEFAULT, TracingMode.TRACING, TracingMode.NEW_TRACING):
            with self.subTest(tracing_mode=tracing_mode):
                self._assert_export(model, tracing_mode)

    @ignore_warnings(UserWarning)
    def test_operator_shift_all_default_libraries(self):
        class BitwiseOperatorShiftModel(torch.nn.Module):
            def forward(self, x, y):
                return x << y, x >> y, x << 1, x >> 1

        model = BitwiseOperatorShiftModel()
        for tracing_mode in (TracingMode.DEFAULT, TracingMode.TRACING, TracingMode.NEW_TRACING):
            with self.subTest(tracing_mode=tracing_mode):
                self._assert_export(model, tracing_mode)


if __name__ == "__main__":
    unittest.main(verbosity=2)
