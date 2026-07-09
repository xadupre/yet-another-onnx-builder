"""
Unit tests for :class:`yobx.torch.in_transformers.YobxOnnxExporter`.

Covers:

* Basic instantiation and attribute access.
* Round-trip export of a tiny ``torch.nn.Linear`` wrapped in a
  ``torch.nn.Module`` when transformers and torch are available.
"""

import unittest

from yobx.ext_test_case import ExtTestCase, requires_torch, requires_transformers


class TestYobxOnnxExporterImport(ExtTestCase):
    @requires_torch("")
    def test_import_yobx_onnx_exporter(self):
        """Importing YobxOnnxExporter should not raise when torch is available."""
        from yobx.torch.in_transformers import YobxOnnxExporter  # noqa: F401

    @requires_transformers("")
    def test_yobx_onnx_exporter_is_subclass_of_onnx_exporter(self):
        """YobxOnnxExporter should be a subclass of transformers' OnnxExporter."""
        from transformers.exporters.exporter_onnx import OnnxExporter

        from yobx.torch.in_transformers import YobxOnnxExporter

        self.assertTrue(issubclass(YobxOnnxExporter, OnnxExporter))

    @requires_transformers("")
    def test_yobx_onnx_exporter_instantiation(self):
        """YobxOnnxExporter can be instantiated without arguments."""
        from yobx.torch.in_transformers import YobxOnnxExporter

        exporter = YobxOnnxExporter()
        self.assertIsNotNone(exporter)

    @requires_transformers("")
    def test_yobx_onnx_exporter_instantiation_with_opset(self):
        """YobxOnnxExporter accepts a target_opset argument."""
        from yobx.torch.in_transformers import YobxOnnxExporter

        exporter = YobxOnnxExporter(target_opset=18)
        self.assertEqual(exporter._target_opset, 18)

    @requires_transformers("")
    def test_yobx_onnx_exporter_required_packages_no_onnxscript(self):
        """YobxOnnxExporter must not list onnxscript in required_packages."""
        from yobx.torch.in_transformers import YobxOnnxExporter

        exporter = YobxOnnxExporter()
        self.assertNotIn("onnxscript", exporter.required_packages)
        self.assertIn("onnx", exporter.required_packages)
        self.assertIn("torch", exporter.required_packages)


class TestYobxOnnxExporterExport(ExtTestCase):
    @requires_transformers("4.0")
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

    @requires_transformers("4.0")
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

    @requires_transformers("4.0")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
