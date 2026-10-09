"""Converts JAX callables into native onnx-light graphs."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
import numpy as np

from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder
from ._lowerings import DynamicSources, lower_jaxpr


def to_onnx(
    model: Callable,
    args: Tuple[Any, ...],
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Tuple[Dict[int, str], ...]] = None,
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    builder_cls: Union[type, Callable] = GraphBuilder,
    verbose: int = 0,
    extra_converters: Optional[Dict[str, Callable]] = None,
    large_model: bool = False,
    external_threshold: int = 1024,
    filename: Optional[str] = None,
    return_optimize_report: bool = False,
) -> ExportArtifact:
    """Converts a JAX callable directly into an onnx-light export artifact.

    Args:
        model: JAX callable accepting positional array inputs.
        args: Concrete sample inputs used to trace the JAX computation.
        input_names: Optional ONNX input names.
        dynamic_shapes: Per-input mappings from axes to symbolic dimension names.
            By default, axis zero of each non-scalar input is named ``batch``.
        target_opset: ONNX opset version or domain-to-version mapping.
        builder_cls: Native graph builder class.
        verbose: Builder verbosity.
        extra_converters: Reserved for signature compatibility. It must be ``None``.
        large_model: Enables external tensor storage.
        external_threshold: Minimum number of elements stored externally.
        filename: Optional path to save the resulting artifact.
        return_optimize_report: Includes native optimization statistics.

    Returns:
        An export artifact containing an onnx-light model.
    """
    arrays = tuple(np.asarray(arg) for arg in args)
    if not arrays:
        raise ValueError("JAX conversion requires at least one input.")
    if input_names is None:
        input_names = ["X" if len(arrays) == 1 else f"X{i}" for i in range(len(arrays))]
    if (
        len(input_names) != len(arrays)
        or any(not isinstance(name, str) or not name for name in input_names)
        or len(set(input_names)) != len(input_names)
    ):
        raise ValueError("input_names must contain one distinct name for every input.")
    if dynamic_shapes is not None and len(dynamic_shapes) != len(arrays):
        raise ValueError("dynamic_shapes must contain one axis mapping for every input.")
    if extra_converters is not None:
        raise ValueError("extra_converters is reserved and must be None for JAX conversion.")
    opsets = {"": target_opset} if isinstance(target_opset, int) else dict(target_opset)
    opsets.setdefault("", DEFAULT_TARGET_OPSET)
    closed = jax.make_jaxpr(model)(*arrays)
    for array, var in zip(arrays, closed.jaxpr.invars):
        aval: Any = var.aval
        if array.dtype != np.dtype(aval.dtype):
            raise TypeError(
                f"JAX traced input dtype {aval.dtype} differs from sample dtype "
                f"{array.dtype}; enable the required JAX dtype before converting."
            )
    builder = builder_cls(opsets, verbose=verbose)
    dynamic_sources: DynamicSources = {}
    static_sizes: set[int] = set()
    for i, (name, array) in enumerate(zip(input_names, arrays)):
        shape = list(array.shape)
        axes = ({0: "batch"} if shape else {}) if dynamic_shapes is None else dynamic_shapes[i]
        static_sizes.update(size for axis, size in enumerate(array.shape) if axis not in axes)
        for axis, dimension in axes.items():
            if (
                not isinstance(axis, int)
                or axis < 0
                or axis >= len(shape)
                or not isinstance(dimension, str)
                or not dimension
            ):
                raise ValueError(f"Invalid dynamic axis {axis!r} for input {name!r}.")
            shape[axis] = dimension
            sample_size = array.shape[axis]
            if sample_size not in dynamic_sources:
                dynamic_sources[sample_size] = (name, axis, dimension)
            else:
                source_info = dynamic_sources[sample_size]
                if source_info is not None and source_info[2] != dimension:
                    dynamic_sources[sample_size] = None
        builder.make_tensor_input(name, np_dtype_to_tensor_dtype(array.dtype), tuple(shape))
    for size in static_sizes.intersection(dynamic_sources):
        dynamic_sources[size] = None
    outputs = set()
    for name in lower_jaxpr(builder, closed, input_names, dynamic_sources):
        if name in outputs:
            name = builder.op.Identity(name)
        builder.make_tensor_output(name, indexed=False, allow_untyped_output=True)
        outputs.add(name)
    artifact = builder.to_onnx(
        large_model=large_model,
        external_threshold=external_threshold,
        return_optimize_report=return_optimize_report,
    )
    if filename:
        artifact.save(filename)
    return artifact
