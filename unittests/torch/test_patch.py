import unittest
import torch
from yobx.ext_test_case import ExtTestCase, requires_torch, hide_stdout
from yobx.torch import apply_patches_for_model
from yobx.torch.in_transformers.patches import enable_transformers_onnx_export_flags
from yobx.helpers.patch_helper import PatchDetails


@requires_torch("2.0")
class TestPatch(ExtTestCase):
    @hide_stdout()
    def test_apply_patches_for_model(self):
        with apply_patches_for_model(patch_torch=True, verbose=1) as details:
            self.assertIsInstance(details, PatchDetails)
            self.assertGreater(len(details), 1)
            patch0 = details[0]
            diff = patch0.make_diff()
            self.assertInOr(
                (
                    "-    if not self.symbol_to_source.get(expr):",
                    "-    assert self.symbol_to_source.get(expr)",
                ),
                diff,
            )

    def test_enable_transformers_onnx_export_flags(self):
        class FakeConfig:
            onnx_export = False

        class FakeSub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.onnx_trace = False

            def prepare_for_onnx_export_(self):
                self.onnx_trace = True

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeConfig()
                self.sub = FakeSub()

        model = FakeModel()
        with enable_transformers_onnx_export_flags(model) as report:
            self.assertTrue(model.config.onnx_export)
            self.assertTrue(model.sub.onnx_trace)
            self.assertEqual(len(report["configs"]), 1)
            self.assertEqual(len(report["modules"]), 1)
        self.assertFalse(model.config.onnx_export)
        self.assertFalse(model.sub.onnx_trace)

    def test_enable_transformers_onnx_export_flags_none(self):
        with enable_transformers_onnx_export_flags(None) as report:
            self.assertEqual(report, {"configs": [], "modules": []})

    def test_apply_patches_for_model_toggles_onnx_export_flags(self):
        class FakeConfig:
            onnx_export = False

        class FakeSub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.onnx_trace = False

            def prepare_for_onnx_export_(self):
                self.onnx_trace = True

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeConfig()
                self.sub = FakeSub()

        model = FakeModel()
        with apply_patches_for_model(patch_transformers=True, model=model):
            self.assertTrue(model.config.onnx_export)
            self.assertTrue(model.sub.onnx_trace)
        self.assertFalse(model.config.onnx_export)
        self.assertFalse(model.sub.onnx_trace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
