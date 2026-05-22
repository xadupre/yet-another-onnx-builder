import unittest
import torch
from yobx.ext_test_case import ExtTestCase, requires_transformers, hide_stdout
from yobx.helpers.patch_helper import PatchDetails
from yobx.torch import (
    apply_patches_for_model,
    use_dyn_not_str,
    register_flattening_functions,
    get_tiny_model,
)


class TestPatchHelper(ExtTestCase):
    @hide_stdout()
    @requires_transformers("4.57")
    def test_patch_details(self):
        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches_for_model(patch_transformers=True, patch_torch=True) as details,
        ):
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
            apply_patches_for_model(patch_transformers=True, verbose=10, patch_torch=True),
        ):
            pass

    @requires_transformers("4.57")
    def test_involved_patches_no_patch_applied(self):
        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        with (
            apply_patches_for_model(patch_transformers=False, patch_torch=False) as details,
            register_flattening_functions(patch_transformers=True),
        ):
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds))
            self.assertIsInstance(details, PatchDetails)
        patches = details.patches_involved_in_graph(ep.graph)
        self.assertEqual(len(patches), 0)

    def test_involved_patches_tiny_broadcast_add(self):
        from yobx.torch.tiny_models import TinyBroadcastAddModel

        model = TinyBroadcastAddModel()
        inputs = TinyBroadcastAddModel._export_inputs()
        dynamic_shapes = use_dyn_not_str(TinyBroadcastAddModel._dynamic_shapes())

        with (
            torch.fx.experimental._config.patch(backed_size_oblivious=True),
            apply_patches_for_model(patch_torch=True) as details,
        ):
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=dynamic_shapes)

        # The graph contains nodes traced from yobx.torch.tiny_models, but none
        # of them come from a registered patch.  Previously this raised
        # ``AssertionError: One node was patched but no patch was found``.
        patches = details.patches_involved_in_graph(ep.graph)
        self.assertIsInstance(patches, list)
        for _patch_info, nodes in patches:
            self.assertIsInstance(nodes, list)
            self.assertGreater(len(nodes), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
