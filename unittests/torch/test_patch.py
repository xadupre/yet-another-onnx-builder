import unittest
import torch
from yobx.ext_test_case import ExtTestCase, requires_torch, hide_stdout
from yobx.torch import apply_patches_for_model, get_tiny_model
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

    def test_add_model_forward(self):
        """get_tiny_model('add') can run a forward pass."""
        data = get_tiny_model("add")
        result = data.model(**data.export_inputs)
        expected = data.export_inputs["x"] + data.export_inputs["y"]
        self.assertTrue(torch.allclose(result, expected))

    def test_add_model_export_with_patch(self):
        """'add' model exports successfully under patch_torch=True (tests
        patched__get_range_constraints and patched_DynamicDimConstraintPrinter)."""
        data = get_tiny_model("add")
        with apply_patches_for_model(patch_torch=True) as details:
            ep = torch.export.export(
                data.model,
                (),
                kwargs=data.export_inputs,
                dynamic_shapes=data.dynamic_shapes,
            )
            self.assertIsInstance(details, PatchDetails)
            self.assertGreater(len(details), 0)
        got = ep.module()(**data.export_inputs)
        expected = data.model(**data.export_inputs)
        self.assertTrue(torch.allclose(got, expected))

    def test_broadcast_multiply_model_forward(self):
        """get_tiny_model('broadcast_multiply') can run a forward pass."""
        data = get_tiny_model("broadcast_multiply")
        result = data.model(**data.export_inputs)
        expected = data.export_inputs["x"] * data.export_inputs["w"]
        self.assertTrue(torch.allclose(result, expected))

    def test_broadcast_multiply_model_export_with_patch(self):
        """'broadcast_multiply' model exports successfully under patch_torch=True (tests
        patched_infer_size and patched__broadcast_shapes)."""
        data = get_tiny_model("broadcast_multiply")
        with apply_patches_for_model(patch_torch=True) as details:
            ep = torch.export.export(
                data.model,
                (),
                kwargs=data.export_inputs,
                dynamic_shapes=data.dynamic_shapes,
            )
            self.assertIsInstance(details, PatchDetails)
            self.assertGreater(len(details), 0)
        got = ep.module()(**data.export_inputs)
        expected = data.model(**data.export_inputs)
        self.assertTrue(torch.allclose(got, expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
