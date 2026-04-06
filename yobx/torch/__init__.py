from typing import Any
import torch
from .export_options import ExportOptions, TracingMode, ConvertingLibrary
from .flatten import register_flattening_functions
from .input_observer import InputObserver
from .interpreter import to_onnx, FunctionOptions, Dispatcher, ForceDispatcher
from .patch import apply_patches_for_model
from .tracing import (
    CustomProxy,
    CustomAttribute,
    CustomParameterProxy,
    CustomProxyBool,
    CustomProxyInt,
    CustomProxyFloat,
    CustomProxyShape,
    CustomTracer,
)
from .tiny_models import get_tiny_model
from .validate import validate_model, DEFAULT_PROMPT, ValidateSummary, ValidateData


def use_dyn_not_str(dynamic_shapes: Any, default_value=None) -> Any:
    """
    Some functions return dynamic shapes as string.
    This function replaces them with ``torch.export.Dim.DYNAMIC``.
    ``default_value=torch.export.Dim.AUTO`` changes the default value.
    """
    if isinstance(dynamic_shapes, list):
        return [use_dyn_not_str(a, default_value=default_value) for a in dynamic_shapes]
    if isinstance(dynamic_shapes, tuple):
        return tuple(use_dyn_not_str(a, default_value=default_value) for a in dynamic_shapes)
    if isinstance(dynamic_shapes, dict):
        return {
            k: use_dyn_not_str(v, default_value=default_value) for k, v in dynamic_shapes.items()
        }
    if isinstance(dynamic_shapes, set):
        return {use_dyn_not_str(a, default_value=default_value) for a in dynamic_shapes}
    if isinstance(dynamic_shapes, str):
        return torch.export.Dim.DYNAMIC if default_value is None else default_value
    return dynamic_shapes
