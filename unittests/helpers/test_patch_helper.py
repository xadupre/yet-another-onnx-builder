import unittest
import torch
from yobx.ext_test_case import (
    ExtTestCase,
    requires_transformers,
    hide_stdout,
    has_transformers,
)
from yobx.helpers.patch_helper import PatchDetails
from yobx.torch import apply_patches
from yobx.torch import use_dyn_not_str
from yobx.torch import register_flattening_functions, get_tiny_model


class TestPatchHelper(ExtTestCase):
    @hide_stdout()
    @requires_transformers("4.57")
    def test_patch_details(self):
        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches(patch_transformers=True) as details,
        ):
            pass
        self.assertGreater(details.n_patches, 1)
        data = details.data()
        self.assertEqual(len(data), details.n_patches)
        for patch in details.patched:
            _kind, f1, f2 = patch.family, patch.function_to_patch, patch.patch
            raw = patch.format_diff(format="raw")
            if callable(f1):
                self.assertIn(f1.__name__, raw)
            self.assertIn(f2.__name__, raw)
            rst = patch.format_diff(format="rst")
            self.assertIn("====", rst)

        # second time to make every patch was removed
        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches(
                patch_transformers=True,
                verbose=10,
                patch_torch=True,
                patch_diffusers=True,
                patch_details=details,
            ),
        ):
            pass

    @requires_transformers("4.57")
    def test_involved_patches_no_patch_applied(self):
        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        with (
            apply_patches(patch_transformers=False, patch_torch=False),
            register_flattening_functions(patch_transformers=True) as details,
        ):
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds))
            self.assertIsInstance(details, PatchDetails)
        patches = details.patches_involved_in_graph(ep.graph)
        self.assertEqual(len(patches), 0)

    @requires_transformers("4.57")
    def test_involved_patches_dynamic_rope(self):
        data = get_tiny_model(
            "arnir0/Tiny-LLM",
            config_updates={"rope_parameters": {"rope_theta": 10000.0, "rope_type": "dynamic"}},
        )
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        print(model.config)
        with (
            apply_patches(patch_transformers=False, patch_torch=False),
            register_flattening_functions(patch_transformers=True) as details,
        ):
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds))
        patches = details.patches_involved_in_graph(ep.graph)
        self.assertEqual(len(patches), 0)
        self.assertNotEmpty(patches)

        report = details.make_report(patches, format="rst")
        if has_transformers("4.51"):
            self.assertIn("====", report)
            self.assertIn("def longrope_frequency_update", report)

    @requires_transformers("4.57")
    def test_involved_patches_long_rope(self):
        data = get_tiny_model(
            "arnir0/Tiny-LLM",
            config_updates={"rope_parameters": {"rope_theta": 10000.0, "rope_type": "longrope"}},
        )
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        print(model.config)
        with (
            apply_patches(patch_transformers=False, patch_torch=False),
            register_flattening_functions(patch_transformers=True) as details,
        ):
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds))
        patches = details.patches_involved_in_graph(ep.graph)
        self.assertEqual(len(patches), 0)
        self.assertNotEmpty(patches)

        report = details.make_report(patches, format="rst")
        if has_transformers("4.51"):
            self.assertIn("====", report)
            self.assertIn("def longrope_frequency_update", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
