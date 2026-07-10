"""
Unit tests for :class:`yobx.torch.in_transformers.YobxOnnxExporter`.

Covers:

* Basic instantiation and attribute access.
* Round-trip export of a tiny ``torch.nn.Linear`` wrapped in a
  ``torch.nn.Module`` when transformers and torch are available.
* Export of a real causal LM (``arnir0/Tiny-LLM``) with a
  :class:`transformers.cache_utils.DynamicCache` (past-key-value cache)
  and dynamic shapes, verifying that the resulting ONNX model contains
  symbolic (dynamic) dimensions.
"""

import unittest

from yobx.ext_test_case import ExtTestCase, requires_torch, requires_transformers


class TestYobxOnnxExporterImport(ExtTestCase):
    @requires_torch("")
    def test_import_yobx_onnx_exporter(self):
        """Importing YobxOnnxExporter should not raise when torch is available."""
        from yobx.torch.in_transformers import YobxOnnxExporter  # noqa: F401

    @requires_transformers("5.12")
    def test_yobx_onnx_exporter_is_subclass_of_onnx_exporter(self):
        """YobxOnnxExporter should be a subclass of transformers' OnnxExporter."""
        from transformers.exporters.exporter_onnx import OnnxExporter

        from yobx.torch.in_transformers import YobxOnnxExporter

        self.assertTrue(issubclass(YobxOnnxExporter, OnnxExporter))

    @requires_transformers("5.12")
    def test_yobx_onnx_exporter_instantiation(self):
        """YobxOnnxExporter can be instantiated without arguments."""
        from yobx.torch.in_transformers import YobxOnnxExporter

        exporter = YobxOnnxExporter()
        self.assertIsNotNone(exporter)

    @requires_transformers("5.12")
    def test_yobx_onnx_exporter_instantiation_with_opset(self):
        """YobxOnnxExporter accepts a target_opset argument."""
        from yobx.torch.in_transformers import YobxOnnxExporter

        exporter = YobxOnnxExporter(target_opset=18)
        self.assertEqual(exporter._target_opset, 18)

    @requires_transformers("5.12")
    def test_yobx_onnx_exporter_required_packages_no_onnxscript(self):
        """YobxOnnxExporter must not list onnxscript in required_packages."""
        from yobx.torch.in_transformers import YobxOnnxExporter

        exporter = YobxOnnxExporter()
        self.assertNotIn("onnxscript", exporter.required_packages)
        self.assertIn("onnx", exporter.required_packages)
        self.assertIn("torch", exporter.required_packages)


class TestYobxOnnxExporterExport(ExtTestCase):
    @requires_transformers("5.12")
    def test_export_simple_module(self):
        """Exports a simple torch.nn.Linear wrapped as a transformers-compatible module."""
        import torch
        from transformers import PretrainedConfig, PreTrainedModel
        from transformers.exporters.configs import OnnxConfig

        from yobx.container import ExportArtifact
        from yobx.torch.in_transformers import YobxOnnxExporter

        class TinyConfig(PretrainedConfig):
            model_type = "tiny_linear"

            def __init__(self, in_features: int = 4, out_features: int = 2, **kwargs):
                super().__init__(**kwargs)
                self.in_features = in_features
                self.out_features = out_features

        class TinyLinear(PreTrainedModel):
            config_class = TinyConfig

            def __init__(self, config: TinyConfig):
                super().__init__(config)
                self.linear = torch.nn.Linear(config.in_features, config.out_features, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        config = TinyConfig(in_features=4, out_features=2)
        model = TinyLinear(config).eval()

        x = torch.randn(3, 4)
        sample_inputs = {"x": x}

        exporter = YobxOnnxExporter()
        artifact = exporter.export(model, sample_inputs, config=OnnxConfig(dynamic=False))

        self.assertIsInstance(artifact, ExportArtifact)
        # The exported proto must have at least one node (the linear op).
        self.assertGreater(len(artifact.graph.node), 0)

    @requires_transformers("5.12")
    def test_export_with_target_opset(self):
        """Respects the target_opset constructor argument during export."""
        import torch
        from transformers import PretrainedConfig, PreTrainedModel
        from transformers.exporters.configs import OnnxConfig

        from yobx.torch.in_transformers import YobxOnnxExporter

        class TinyConfig(PretrainedConfig):
            model_type = "tiny_linear_opset"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        class TinyLinear(PreTrainedModel):
            config_class = TinyConfig

            def __init__(self, config):
                super().__init__(config)
                self.linear = torch.nn.Linear(4, 2, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = TinyLinear(TinyConfig()).eval()
        x = torch.randn(3, 4)

        exporter = YobxOnnxExporter(target_opset=18)
        artifact = exporter.export(model, {"x": x}, config=OnnxConfig(dynamic=False))

        self.assertEqual(artifact.opset_import[0].version, 18)

    @requires_transformers("5.12")
    def test_export_with_dict_config(self):
        """Accepts a plain dict as config in addition to OnnxConfig instances."""
        import torch
        from transformers import PretrainedConfig, PreTrainedModel

        from yobx.container import ExportArtifact
        from yobx.torch.in_transformers import YobxOnnxExporter

        class TinyConfig(PretrainedConfig):
            model_type = "tiny_linear_dict"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        class TinyLinear(PreTrainedModel):
            config_class = TinyConfig

            def __init__(self, config):
                super().__init__(config)
                self.linear = torch.nn.Linear(4, 2, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = TinyLinear(TinyConfig()).eval()
        x = torch.randn(3, 4)

        exporter = YobxOnnxExporter()
        artifact = exporter.export(model, {"x": x}, config={"dynamic": False})

        self.assertIsInstance(artifact, ExportArtifact)

    @requires_transformers("5.12")
    def test_export_module_with_dynamic_cache_and_dynamic_shapes(self):
        """Exports a minimal PreTrainedModel with a DynamicCache input and dynamic shapes.

        Verifies end-to-end that symbolic dimensions survive in the ONNX output
        when the model accepts a KV-cache (DynamicCache) as ``past_key_values``
        and ``OnnxConfig(dynamic=True, dynamic_shapes=...)`` is supplied.
        The model and weights are entirely self-contained — no HuggingFace
        checkpoint download is required.
        """
        import torch
        from transformers import PretrainedConfig, PreTrainedModel
        from transformers.exporters.configs import OnnxConfig

        from yobx.container import ExportArtifact
        from yobx.torch import apply_patches_for_model, register_flattening_functions
        from yobx.torch.in_transformers import YobxOnnxExporter
        from yobx.torch.in_transformers.cache_helper import CacheKeyValue, make_dynamic_cache

        # ---- minimal model definition ----------------------------------------

        class TinyConfig(PretrainedConfig):
            model_type = "tiny_cached_lm_unit"

            def __init__(self, vocab_size: int = 32, embed_dim: int = 4, **kwargs):
                super().__init__(**kwargs)
                self.vocab_size = vocab_size
                self.embed_dim = embed_dim

        class TinyCachedLM(PreTrainedModel):
            """Tiny LM that consumes the first layer's key cache as an additive bias."""

            config_class = TinyConfig

            def __init__(self, config: TinyConfig):
                super().__init__(config)
                self.embed = torch.nn.Embedding(config.vocab_size, config.embed_dim)

            def forward(
                self,
                input_ids: torch.Tensor,
                past_key_values=None,
            ) -> torch.Tensor:
                x = self.embed(input_ids)  # (batch, seq, embed_dim)
                if past_key_values is not None:
                    # Reduce the past key over heads and past-seq dims so it
                    # broadcasts cleanly onto x: (batch, 1, embed_dim).
                    k0 = CacheKeyValue(past_key_values).key_cache[0]
                    x = x + k0.mean(dim=(1, 2)).unsqueeze(1)
                return x

        # ---- build a tiny model and a one-layer DynamicCache -----------------

        n_layers = 1
        bsize, nheads, slen, embed_dim = 2, 1, 5, 4

        config = TinyConfig(vocab_size=32, embed_dim=embed_dim)
        model = TinyCachedLM(config).eval()

        past_key_values = make_dynamic_cache(
            [
                (
                    torch.randn(bsize, nheads, slen, embed_dim),
                    torch.randn(bsize, nheads, slen, embed_dim),
                )
                for _ in range(n_layers)
            ]
        )

        sample_inputs = dict(
            input_ids=torch.randint(0, 32, (bsize, 3), dtype=torch.int64),
            past_key_values=past_key_values,
        )

        # Dynamic shapes: batch and sequence length for input_ids; batch and
        # past_length for each of the 2*n_layers flattened cache tensors.
        dynamic_shapes = dict(
            input_ids={0: "batch", 1: "seq_length"},
            past_key_values=[{0: "batch", 2: "past_length"} for _ in range(2 * n_layers)],
        )

        # ---- export ----------------------------------------------------------

        exporter = YobxOnnxExporter()

        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches_for_model(
                patch_torch=True, patch_transformers=True, model=model
            ),
        ):
            artifact = exporter.export(
                model,
                sample_inputs,
                config=OnnxConfig(dynamic=True, dynamic_shapes=dynamic_shapes),
            )

        self.assertIsInstance(artifact, ExportArtifact)
        self.assertGreater(len(artifact.graph.node), 0)

        # At least one ONNX input dimension must carry a symbolic name.
        dynamic_dims = [
            d.dim_param
            for inp in artifact.graph.input
            for d in inp.type.tensor_type.shape.dim
            if d.dim_param
        ]
        self.assertGreater(
            len(dynamic_dims),
            0,
            "The ONNX model must contain at least one dynamic (symbolic) dimension.",
        )


class TestYobxOnnxExporterRealModels(ExtTestCase):
    @requires_transformers("5.12")
    def test_export_tiny_llm_with_cache_and_dynamic_shapes(self):
        """Exports arnir0/Tiny-LLM with a DynamicCache and dynamic shapes.

        Verifies that the resulting ONNX model contains at least one dynamic
        (symbolic) dimension, confirming that dynamic shapes are preserved
        end-to-end through YobxOnnxExporter.
        """
        from transformers.exporters.configs import OnnxConfig

        from yobx.container import ExportArtifact
        from yobx.torch import apply_patches_for_model, register_flattening_functions
        from yobx.torch.in_transformers import YobxOnnxExporter
        from yobx.torch.tiny_models import get_tiny_model
        from yobx.torch.torch_helper import torch_deepcopy

        model_data = get_tiny_model("arnir0/Tiny-LLM")
        exporter = YobxOnnxExporter()

        # register_flattening_functions patches the pytree registration for
        # DynamicCache so that torch.export.export can flatten and unflatten
        # the KV cache correctly.  apply_patches_for_model applies the
        # yobx-level torch and transformers patches needed for tracing.
        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches_for_model(
                patch_torch=True, patch_transformers=True, model=model_data.model
            ),
        ):
            artifact = exporter.export(
                model_data.model,
                torch_deepcopy(model_data.export_inputs),
                config=OnnxConfig(dynamic=True, dynamic_shapes=model_data.dynamic_shapes),
            )

        self.assertIsInstance(artifact, ExportArtifact)
        self.assertGreater(len(artifact.graph.node), 0)

        # Verify that the ONNX model has at least one symbolic (dynamic) dim.
        dynamic_dims = [
            d.dim_param
            for inp in artifact.graph.input
            for d in inp.type.tensor_type.shape.dim
            if d.dim_param
        ]
        self.assertGreater(
            len(dynamic_dims),
            0,
            "The ONNX model must contain at least one dynamic (symbolic) dimension.",
        )

    @requires_transformers("5.12")
    def test_export_tiny_llm_random_weights(self):
        """Exports arnir0/Tiny-LLM with random weights via YobxOnnxExporter."""
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.exporters.configs import OnnxConfig

        from yobx.container import ExportArtifact
        from yobx.torch.in_transformers import YobxOnnxExporter
        from yobx.torch.in_transformers.models import get_cached_configuration

        config = get_cached_configuration("arnir0/Tiny-LLM")
        model = AutoModelForCausalLM.from_config(config).eval()
        sample_inputs = dict(
            input_ids=torch.randint(0, config.vocab_size, (1, 5), dtype=torch.int64),
            attention_mask=torch.ones(1, 5, dtype=torch.int64),
        )

        exporter = YobxOnnxExporter()
        artifact = exporter.export(model, sample_inputs, config=OnnxConfig(dynamic=False))

        self.assertIsInstance(artifact, ExportArtifact)
        self.assertGreater(len(artifact.graph.node), 0)

    @requires_transformers("5.12")
    def test_to_onnx_strategy_transformers_tiny_llm(self):
        """to_onnx with ExportOptions(strategy='transformers') routes through YobxOnnxExporter."""
        import torch
        from transformers import AutoModelForCausalLM

        from yobx.container import ExportArtifact
        from yobx.torch import to_onnx
        from yobx.torch.export_options import ExportOptions
        from yobx.torch.in_transformers.models import get_cached_configuration

        config = get_cached_configuration("arnir0/Tiny-LLM")
        model = AutoModelForCausalLM.from_config(config).eval()
        sample_inputs = dict(
            input_ids=torch.randint(0, config.vocab_size, (1, 5), dtype=torch.int64),
            attention_mask=torch.ones(1, 5, dtype=torch.int64),
        )

        artifact = to_onnx(
            model, kwargs=sample_inputs, export_options=ExportOptions(strategy="transformers")
        )

        self.assertIsInstance(artifact, ExportArtifact)
        self.assertGreater(len(artifact.graph.node), 0)

    @requires_transformers("5.12")
    def test_export_whisper_tiny_random_weights(self):
        """Exports openai/whisper-tiny with random weights via YobxOnnxExporter."""
        from transformers.exporters.configs import OnnxConfig

        from yobx.container import ExportArtifact
        from yobx.torch.in_transformers import YobxOnnxExporter
        from yobx.torch.tiny_models import get_tiny_model

        model_data = get_tiny_model("openai/whisper-tiny")

        exporter = YobxOnnxExporter()
        artifact = exporter.export(
            model_data.model, model_data.export_inputs, config=OnnxConfig(dynamic=False)
        )

        self.assertIsInstance(artifact, ExportArtifact)
        self.assertGreater(len(artifact.graph.node), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
