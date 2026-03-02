import unittest
import torch
import transformers
from yobx.ext_test_case import ExtTestCase, requires_transformers, has_transformers
from yobx.helpers.patch_helper import PatchDetails
from yobx.torch import (
    apply_patches_for_model,
    use_dyn_not_str,
    register_flattening_functions,
    get_tiny_model,
)
from yobx.torch.torch_helper import torch_deepcopy


class TestPatchTransformerHelper(ExtTestCase):
    def test_is_wrapped(self):
        llama = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
        self.assertTrue(hasattr(llama.forward, "__wrapped__"))
        self.assertTrue(hasattr(llama.forward.__wrapped__, "__code__"))
        s = str(llama.forward.__wrapped__.__code__)
        self.assertIn("transformers/modeling_rope_utils.py", s)

    @requires_transformers("4.57")
    def test_involved_patches_no_patch_applied(self):
        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        expected = model(**torch_deepcopy(inputs))
        with (
            apply_patches_for_model(patch_transformers=False, patch_torch=False) as details,
            register_flattening_functions(patch_transformers=True),
        ):
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds))
            self.assertIsInstance(details, PatchDetails)
            patches = details.patches_involved_in_graph(ep.graph)
            self.assertEqual(len(patches), 0)
            got = ep.module(**inputs)
            self.assertEqualAny(expected, got)

    @requires_transformers("4.57")
    @unittest.skip("dynamic_rope does not work with this model")
    def test_involved_patches_dynamic_rope(self):
        data = get_tiny_model(
            "arnir0/Tiny-LLM",
            config_updates={
                "rope_parameters": {
                    "factor": 10.0,
                    "rope_type": "dynamic",
                    "partial_rotary_factor": 0.4,
                },
                "head_dim": 38,
            },
        )
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        expected = model(**torch_deepcopy(inputs))
        with (
            apply_patches_for_model(
                patch_transformers=True, patch_torch=False, verbose=10, model=model
            ) as details,
            register_flattening_functions(patch_transformers=True),
        ):
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds))
            patches = details.patches_involved_in_graph(ep.graph)
            self.assertEqual(len(patches), 0)
            self.assertNotEmpty(patches)

            report = details.make_report(patches, format="rst")
            if has_transformers("4.51"):
                self.assertIn("====", report)
                self.assertIn("def dynamic_frequency_update", report)
            got = ep.module(**inputs)
            self.assertEqualAny(expected, got)

    @requires_transformers("4.57")
    def test_involved_patches_long_rope(self):
        data = get_tiny_model(
            "arnir0/Tiny-LLM",
            config_updates={
                "rope_parameters": {
                    "long_factor": 12.0,
                    "factor": 10.0,
                    "short_factor": 8.0,
                    "rope_type": "longrope",
                }
            },
        )
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        expected = model(**torch_deepcopy(inputs))
        with (
            apply_patches_for_model(
                patch_transformers=True, patch_torch=False, model=model, verbose=10
            ) as details,
            register_flattening_functions(patch_transformers=True),
        ):
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds))
            patches = details.patches_involved_in_graph(ep.graph)
            self.assertEqual(len(patches), 1)
            self.assertNotEmpty(patches)

            report = details.make_report(patches, format="rst")
            if has_transformers("4.51"):
                self.assertIn("====", report)
                self.assertIn("def longrope_frequency_update", report)
            got = ep.module(**inputs)
            self.assertEqualAny(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
