"""
Provides :class:`YobxOnnxExporter`, a subclass of
:class:`transformers.exporters.OnnxExporter` that replaces the default
``torch.onnx.export`` / onnxscript backend with the yobx converter.

This lets callers use the standard transformers exporter API while benefiting
from yobx's graph-builder optimisations and operator coverage.

Usage::

    from transformers.exporters import OnnxConfig
    from yobx.torch.in_transformers import YobxOnnxExporter

    exporter = YobxOnnxExporter()
    artifact = exporter.export(model, sample_inputs, config=OnnxConfig(dynamic=True))
    # artifact is an ExportArtifact – save it, inspect the proto, etc.
    artifact.save("model.onnx")
"""

from __future__ import annotations

import copy
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.exporters.configs import OnnxConfig
    from yobx.container import ExportArtifact

try:
    from transformers.exporters.exporter_onnx import OnnxExporter as _OnnxExporterBase

    _transformers_available = True
except ImportError:
    _OnnxExporterBase = object  # type: ignore[assignment,misc]
    _transformers_available = False


class YobxOnnxExporter(_OnnxExporterBase):  # type: ignore[valid-type]
    """
    Subclass of :class:`transformers.exporters.OnnxExporter` that converts a
    :class:`~transformers.PreTrainedModel` to ONNX using the **yobx** graph
    builder instead of the default ``torch.onnx.export`` / onnxscript pipeline.

    When ``transformers`` is not installed, instantiating this class raises
    :class:`ImportError`.

    The interface is identical to the upstream exporter:

    .. code-block:: python

        from transformers.exporters import OnnxConfig
        from yobx.torch.in_transformers import YobxOnnxExporter

        exporter = YobxOnnxExporter()
        artifact = exporter.export(model, inputs, config=OnnxConfig(dynamic=True))
        artifact.save("model.onnx")

    Extra keyword arguments passed to the constructor are forwarded verbatim
    to :func:`yobx.torch.to_onnx` during export (e.g. ``options``,
    ``dispatcher``, ``export_modules_as_functions``, …).

    :param target_opset: ONNX opset version to target.  Overrides the value
        carried by ``config.opset_version`` when both are supplied.
    :param kwargs: extra keyword arguments forwarded verbatim to
        :func:`yobx.torch.to_onnx`.
    """

    # yobx does not use onnxscript — remove it from the required-packages
    # list so that the environment check in HfExporter.__init__ does not
    # fail when onnxscript is absent.
    required_packages: List[str] = ["torch", "onnx"]
    tested_versions: Dict[str, str] = {"torch": "2.4.0", "onnx": "1.16.0"}

    def __init__(self, target_opset: Optional[int] = None, **kwargs: Any) -> None:
        if not _transformers_available:
            raise ImportError(
                "YobxOnnxExporter requires the 'transformers' package. "
                "Install it with: pip install transformers"
            )
        self._target_opset = target_opset
        self._extra_kwargs: Dict[str, Any] = kwargs
        # Call HfExporter.__init__ which validates required_packages.
        super().__init__()

    def export(  # type: ignore[override]
        self,
        model: "PreTrainedModel",
        sample_inputs: MutableMapping[str, Any],
        config: Union["OnnxConfig", Dict[str, Any]],
    ) -> "ExportArtifact":
        """
        Exports *model* to ONNX using the yobx converter.

        Applies the same transformers-side preprocessing as
        :class:`transformers.exporters.DynamoExporter` (label stripping,
        output-flag patching, dynamic-shape inference) and then feeds the
        resulting :class:`torch.export.ExportedProgram` to
        :func:`yobx.torch.to_onnx`.

        :param model: the :class:`~transformers.PreTrainedModel` to export.
        :param sample_inputs: forward kwargs — what you would pass to
            ``model(**sample_inputs)``.  Labels and loss-related keys must
            *not* be present (see
            :class:`transformers.exporters.DynamoExporter`).
        :param config: an :class:`~transformers.exporters.OnnxConfig` (or a
            plain :class:`dict` that will be converted to one) controlling
            dynamic shapes, output path, opset version, etc.
        :return: :class:`~yobx.container.ExportArtifact` wrapping the
            exported ONNX proto.
        """
        if not _transformers_available:
            raise ImportError(
                "YobxOnnxExporter requires the 'transformers' package. "
                "Install it with: pip install transformers"
            )

        import torch
        from transformers.exporters.configs import DynamoConfig, OnnxConfig
        from transformers.exporters.exporter_dynamo import (
            apply_patches,
            get_auto_dynamic_shapes,
            patch_forward_signature,
            patch_model_config,
            prepare_for_export,
            register_cache_pytrees_for_model,
            reset_model_state,
        )

        from transformers import PreTrainedModel as _PreTrainedModel

        from yobx.torch import to_onnx, use_dyn_not_str

        # Normalise config.
        if isinstance(config, dict):
            config = OnnxConfig(**config)
        elif not isinstance(config, DynamoConfig):
            raise TypeError(f"Expected config to be an OnnxConfig or dict, got {type(config)}")

        # Apply transformers-side pre-processing (strips labels, pops output
        # flags, casts tensors to model dtype/device, etc.).
        model, sample_inputs, output_flags = prepare_for_export(model, sample_inputs)  # type: ignore[assignment]
        assert isinstance(model, _PreTrainedModel)

        # patch_forward_signature requires a plain dict, not a MutableMapping.
        inputs_dict: Dict[str, Any] = dict(sample_inputs)

        dynamic_shapes = config.dynamic_shapes
        if config.dynamic and dynamic_shapes is None:
            dynamic_shapes = get_auto_dynamic_shapes(inputs_dict)

        # torch.export.export requires Dim objects, not plain strings.
        # use_dyn_not_str converts any string dimension markers to
        # torch.export.Dim.DYNAMIC so the caller can use either form.
        if dynamic_shapes is not None:
            dynamic_shapes = use_dyn_not_str(dynamic_shapes)

        register_cache_pytrees_for_model(model)

        # Trace the model into an ExportedProgram using the dynamo pipeline.
        # We use apply_patches("dynamo") (not "onnx") because yobx does not
        # require onnxscript-specific graph rewrites.
        with (
            apply_patches("dynamo"),
            reset_model_state(model),
            patch_model_config(model, output_flags),
            patch_forward_signature(model, inputs_dict),
        ):
            exported_program = torch.export.export(
                model,
                args=(),
                kwargs=copy.deepcopy(inputs_dict),
                strict=config.strict,
                dynamic_shapes=dynamic_shapes,
                prefer_deferred_runtime_asserts_over_guards=(
                    config.prefer_deferred_runtime_asserts_over_guards
                ),
            )

        # Determine the target opset: explicit argument wins over config.
        target_opset = self._target_opset
        if target_opset is None and isinstance(config, OnnxConfig) and config.opset_version:
            target_opset = config.opset_version

        # Determine the output filename from config.
        filename: Optional[str] = None
        if isinstance(config, OnnxConfig) and config.output_path is not None:
            filename = str(config.output_path)

        export_kwargs: Dict[str, Any] = dict(self._extra_kwargs)
        if target_opset is not None:
            export_kwargs.setdefault("target_opset", target_opset)
        if filename is not None:
            export_kwargs.setdefault("filename", filename)

        # Convert the ExportedProgram to ONNX using the yobx graph builder.
        artifact = to_onnx(exported_program, **export_kwargs)
        return artifact
