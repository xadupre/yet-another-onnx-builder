from typing import Any
import torch
from .export_options import ExportOptions
from .flatten_helper import register_flattening_functions
from .tracing import (
    CustomProxy,
    CustomAttribute,
    CustomParameterProxy,
    CustomProxyInt,
    CustomProxyFloat,
    CustomTracer,
    CondCCOp,
    LEAVE_INPLACE,
    _len,
    _isinstance,
    replace_problematic_function_before_tracing,
    setitem_with_transformation,
    tree_unflatten_with_proxy,
)


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
