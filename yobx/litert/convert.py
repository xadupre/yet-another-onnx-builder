"""Main entry point for the LiteRT/TFLite → ONNX converter."""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
from ..xbuilder import GraphBuilder, OptimizationOptions
from .litert_helper import BuiltinOperator, TFLiteOperator, TFLiteSubgraph, parse_tflite_model
from .register import get_litert_op_converter


def to_onnx(
    model: Union[str, "os.PathLike[str]", bytes],
    args: Union[Tuple[Any, ...], Sequence[Tuple[Any, ...]]] = (),
    input_names: Optional[Union[Sequence[str], Sequence[Optional[Sequence[str]]]]] = None,
    dynamic_shapes: Optional[
        Union[Tuple[Dict[int, str], ...], Sequence[Optional[Tuple[Dict[int, str], ...]]]]
    ] = None,
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    builder_cls: Union[type, Callable] = GraphBuilder,
    verbose: int = 0,
    extra_converters: Optional[Dict[int, Callable]] = None,
    large_model: bool = False,
    external_threshold: int = 1024,
    subgraph_index: Optional[int] = 0,
    return_optimize_report: bool = False,
) -> ExportArtifact:
    """Convert a :epkg:`TFLite`/:epkg:`LiteRT` model to ONNX.

    The function parses the binary FlatBuffer that every ``.tflite`` file
    uses and walks every operator in the requested subgraph.  Each operator
    is converted to its ONNX equivalent by a registered converter
    (see :mod:`yobx.litert.register`).

    :param model: path to a ``.tflite`` file **or** raw model bytes
    :param args: dummy inputs used to determine dtypes and shapes when the
        model does not carry that information.  For a single subgraph (the
        default) this is a :class:`tuple` of numpy arrays **or**
        :class:`onnx.ValueInfoProto` descriptors.  When *subgraph_index* is
        ``None`` (merge all subgraphs) this must be a sequence of such
        tuples, one per subgraph; use an empty sequence ``()`` or ``[]`` to
        let the converter infer all types from the model's own tensor metadata.
    :param input_names: optional name(s) for the ONNX input tensors.  For a
        single subgraph: a :class:`~typing.Sequence` of strings.  When
        *subgraph_index* is ``None``: a sequence of per-subgraph name lists
        (or ``None`` entries to use the TFLite tensor names for that subgraph).
    :param dynamic_shapes: optional per-input axis-to-dim-name mappings.
        When *None*, axis 0 is treated as a dynamic batch dimension.  When
        *subgraph_index* is ``None``: a sequence of per-subgraph mappings.
    :param target_opset: opset version; either an integer for the default
        domain (``""``) or a ``Dict[str, int]`` mapping domain names to
        versions.
    :param builder_cls: by default the graph builder is a
        :class:`yobx.xbuilder.GraphBuilder` but any builder can be used
        as long it implements the :ref:`builder-api`.
    :param verbose: verbosity level (0 = silent).
    :param extra_converters: optional mapping from
        :class:`~yobx.litert.litert_helper.BuiltinOperator` int (or custom-op
        name string) to converter function with signature
        ``(g, sts, outputs, op)``.  Entries here take priority over the
        built-in op converters.
    :param large_model: if *True* the returned
        :class:`~yobx.container.ExportArtifact` has its
        :attr:`~yobx.container.ExportArtifact.container` attribute set to an
        :class:`~yobx.container.ExtendedModelContainer`.
    :param external_threshold: if *large_model* is *True*, every tensor
        whose element count exceeds this threshold is stored as external data.
    :param subgraph_index: index of the subgraph to convert (default: 0).
        Pass ``None`` to **merge all** subgraphs into a single ONNX model.
        Tensor names are automatically prefixed with ``sg{i}_`` (where *i* is
        the subgraph index) to avoid name collisions across subgraphs.
    :param return_optimize_report: if True, the returned
        :class:`~yobx.container.ExportArtifact` has its
        :attr:`~yobx.container.ExportArtifact.report` attribute populated with
        per-pattern optimization statistics
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX proto together with an :class:`~yobx.container.ExportReport`.

    Example — single subgraph::

        import numpy as np
        from yobx.litert import to_onnx

        X = np.random.rand(1, 4).astype(np.float32)
        artifact = to_onnx("model.tflite", (X,))
        proto = artifact.proto
        artifact.save("model.onnx")

    Example — merge all subgraphs::

        artifact = to_onnx("model.tflite", subgraph_index=None)
        artifact.save("merged.onnx")
    """
    from . import register_litert_converters

    if isinstance(target_opset, int):
        dict_target_opset: Dict[str, int] = {"": target_opset}
    else:
        if not isinstance(target_opset, dict):
            raise TypeError(f"target_opset must be an int or a dict, not {type(target_opset)}")
        dict_target_opset = target_opset.copy()
        if "" not in dict_target_opset:
            dict_target_opset[""] = DEFAULT_TARGET_OPSET

    register_litert_converters()

    tflite_model = parse_tflite_model(model)

    kwargs: Dict[str, Any] = {}
    if "com.microsoft" in dict_target_opset:
        kwargs["optimization_options"] = OptimizationOptions(patterns="default+onnxruntime")
    if verbose and issubclass(builder_cls, GraphBuilder):  # type: ignore
        kwargs["verbose"] = verbose

    if subgraph_index is None:
        # Merge all subgraphs into a single ONNX model.
        # Tensor names are prefixed with "sg{i}_" to avoid cross-subgraph
        # name collisions (except when there is only one subgraph).
        g = builder_cls(dict_target_opset, **kwargs)
        n_subgraphs = len(tflite_model.subgraphs)
        for i, subgraph in enumerate(tflite_model.subgraphs):
            prefix = f"sg{i}_" if n_subgraphs > 1 else ""
            # Resolve per-subgraph args / input_names / dynamic_shapes.
            sg_args: Tuple[Any, ...] = args[i] if i < len(args) else ()  # type: ignore[index]
            sg_input_names: Optional[Sequence[str]] = (
                input_names[i] if input_names is not None and i < len(input_names) else None  # type: ignore[index]
            )
            sg_dynamic_shapes: Optional[Tuple[Dict[int, str], ...]] = (
                dynamic_shapes[i]  # type: ignore
                if dynamic_shapes is not None and i < len(dynamic_shapes)  # type: ignore[arg-type]
                else None
            )
            if sg_input_names is not None and len(sg_input_names) != len(subgraph.inputs):
                raise ValueError(
                    f"Length mismatch for subgraph {i}: model has "
                    f"{len(subgraph.inputs)} input(s) but input_names[{i}]="
                    f"{list(sg_input_names)!r} has {len(sg_input_names)} entries."
                )
            _convert_subgraph(
                subgraph=subgraph,
                g=g,
                args=sg_args,
                input_names=sg_input_names,
                dynamic_shapes=sg_dynamic_shapes,
                verbose=verbose,
                extra_converters=extra_converters or {},
                name_prefix=prefix,
            )
        return _finalize_builder(
            g, large_model, external_threshold, verbose, return_optimize_report
        )

    # --- Single-subgraph path (original behaviour) ---
    if subgraph_index >= len(tflite_model.subgraphs):
        raise ValueError(
            f"subgraph_index={subgraph_index} is out of range; "
            f"the model has {len(tflite_model.subgraphs)} subgraph(s)."
        )
    subgraph = tflite_model.subgraphs[subgraph_index]

    # In the single-subgraph case args/input_names/dynamic_shapes are used
    # directly (not as sequences of per-subgraph values).
    single_args: Tuple[Any, ...] = args  # type: ignore[assignment]
    single_input_names: Optional[Sequence[str]] = input_names  # type: ignore[assignment]
    single_dynamic_shapes: Optional[Tuple[Dict[int, str], ...]] = dynamic_shapes  # type: ignore[assignment]

    if single_input_names is not None and len(single_input_names) != len(subgraph.inputs):
        raise ValueError(
            f"Length mismatch: model has {len(subgraph.inputs)} input(s) "
            f"but input_names={single_input_names!r} has {len(single_input_names)} entries."
        )

    g = builder_cls(dict_target_opset, **kwargs)

    _convert_subgraph(
        subgraph=subgraph,
        g=g,
        args=single_args,
        input_names=single_input_names,
        dynamic_shapes=single_dynamic_shapes,
        verbose=verbose,
        extra_converters=extra_converters or {},
    )

    return _finalize_builder(g, large_model, external_threshold, verbose, return_optimize_report)


def _finalize_builder(
    g: Any,
    large_model: bool,
    external_threshold: int,
    verbose: int,
    return_optimize_report: bool = False,
) -> ExportArtifact:
    """Call ``g.to_onnx(...)`` and return the resulting :class:`ExportArtifact`."""
    if isinstance(g, GraphBuilder):
        onx = g.to_onnx(  # type: ignore
            large_model=large_model,
            external_threshold=external_threshold,
            return_optimize_report=return_optimize_report,
        )
        if verbose and onx.report and onx.report.stats:
            import pandas

            df = pandas.DataFrame(onx.report.stats)
            for c in ["added", "removed"]:
                df[c] = df[c].fillna(0).astype(int)
            agg = df.groupby("pattern")[["added", "removed", "time_in"]].sum()
            agg = agg[(agg["added"] > 0) | (agg["removed"] > 0)].sort_values(
                "removed", ascending=False
            )
            if agg.shape[0]:
                print(agg.to_string())
        return onx
    return g.to_onnx(large_model=large_model, external_threshold=external_threshold)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _onnx_dtype_from_arg(arg: Any) -> int:
    """Return the ONNX ``TensorProto`` dtype integer for a numpy array or VIP."""
    if isinstance(arg, onnx.ValueInfoProto):
        return arg.type.tensor_type.elem_type
    arr = np.asarray(arg)
    return _onnx_elem_type_from_np(arr.dtype)


def _shape_from_arg(arg: Any) -> Tuple[Union[int, str], ...]:
    """Return the shape of a numpy array or ValueInfoProto."""
    if isinstance(arg, onnx.ValueInfoProto):
        tt = arg.type.tensor_type
        if tt.HasField("shape"):
            return tuple(d.dim_param if d.dim_param else (d.dim_value or 0) for d in tt.shape.dim)
        return (0,)
    return tuple(np.asarray(arg).shape)


def _convert_subgraph(
    subgraph: TFLiteSubgraph,
    g: GraphBuilder,
    args: Tuple[Any, ...],
    input_names: Optional[Sequence[str]],
    dynamic_shapes: Optional[Tuple[Dict[int, str], ...]],
    verbose: int,
    extra_converters: Dict[int, Callable],
    name_prefix: str = "",
) -> None:
    """Walk the TFLite subgraph and populate the ONNX GraphBuilder.

    :param subgraph: parsed TFLite subgraph
    :param g: ONNX graph builder
    :param args: dummy inputs for dtype/shape hints
    :param input_names: optional override for input tensor names
    :param dynamic_shapes: optional dynamic axis specifications
    :param verbose: verbosity level
    :param extra_converters: op-code → converter overrides
    :param name_prefix: string prepended to every tensor name produced by this
        subgraph; used when merging multiple subgraphs to avoid name collisions.
    """
    tensors = subgraph.tensors

    # ------------------------------------------------------------------ #
    # 1. Register model weights as ONNX initializers.                     #
    #    All tensors that have buffer data and are not graph inputs are    #
    #    treated as constant initializers.                                 #
    # ------------------------------------------------------------------ #
    input_set = set(subgraph.inputs)
    _output_set = set(subgraph.outputs)

    for t in tensors:
        if t.data is not None and t.index not in input_set:
            prefixed = name_prefix + t.name
            if not g.has_name(prefixed):
                g.make_initializer(prefixed, t.data, source="_convert_subgraph.weight")

    # ------------------------------------------------------------------ #
    # 2. Register ONNX inputs.                                            #
    # ------------------------------------------------------------------ #
    for arg_pos, t_idx in enumerate(subgraph.inputs):
        tensor = tensors[t_idx]

        # Internal (graph-builder) name for the tensor — always prefixed so
        # that ops can look it up by the same key regardless of the path.
        internal_name = name_prefix + tensor.name

        # The ONNX graph input name: user-supplied or the prefixed tensor name.
        if input_names is not None:
            name = input_names[arg_pos]
        else:
            name = internal_name

        # Determine dtype.
        if arg_pos < len(args):
            arg = args[arg_pos]
            if isinstance(arg, onnx.ValueInfoProto):
                elem_type = arg.type.tensor_type.elem_type
            else:
                elem_type = _onnx_elem_type_from_np(np.asarray(arg).dtype)
        else:
            elem_type = _tflite_dtype_to_onnx_elem_type(tensor.dtype)

        # Determine shape.
        shape_list = list(tensor.shape)
        if dynamic_shapes and arg_pos < len(dynamic_shapes):
            for axis, dim_name in dynamic_shapes[arg_pos].items():
                if axis < len(shape_list):
                    shape_list[axis] = dim_name  # type: ignore
        elif shape_list:
            # Default: make axis 0 dynamic.
            shape_list[0] = "batch"  # type: ignore

        shape_tuple: Tuple[Union[int, str], ...] = tuple(shape_list)

        if not g.has_name(name):
            g.make_tensor_input(name, elem_type, shape_tuple)

        # Alias the ONNX input name → internal (prefixed) tensor name so
        # downstream ops that look up the original TFLite tensor name can
        # find it through the builder.
        if name != internal_name and not g.has_name(internal_name):
            g.op.Identity(name, outputs=[internal_name], name=f"{name_prefix}litert_input_alias")

    # ------------------------------------------------------------------ #
    # 3. Convert operators in topological order (TFLite guarantees this). #
    # ------------------------------------------------------------------ #
    for op in subgraph.operators:
        # Build input names: skip tensors with index -1 (optional absent).
        op_input_names = [(name_prefix + tensors[i].name) if i >= 0 else "" for i in op.inputs]
        op_output_names = [name_prefix + tensors[i].name for i in op.outputs if i >= 0]

        # Resolve converter: extra_converters > registry.
        fct = extra_converters.get(op.opcode)
        if fct is None and op.opcode == BuiltinOperator.CUSTOM:
            fct = extra_converters.get(op.custom_code)  # type: ignore
        if fct is None:
            fct = get_litert_op_converter(op.opcode)
        if fct is None and op.opcode == BuiltinOperator.CUSTOM:
            fct = get_litert_op_converter(op.custom_code)

        if fct is None:
            raise RuntimeError(
                f"No converter registered for TFLite op {op.name!r} "
                f"(opcode={op.opcode}).  "
                f"Provide a converter via extra_converters or "
                f"implement one in yobx/litert/ops/."
            )

        # Build a proxy op object so converters can access inputs/outputs
        # by tensor name (as strings) rather than index.
        proxy = _OpProxy(op, op_input_names, op_output_names)

        sts: Dict[str, Any] = {}
        if verbose >= 2:
            print(
                f"[yobx.litert] converting op {op.name!r} "
                f"inputs={op_input_names} outputs={op_output_names}"
            )
        fct(g, sts, op_output_names, proxy)

    # ------------------------------------------------------------------ #
    # 4. Register ONNX outputs.                                           #
    # ------------------------------------------------------------------ #
    for t_idx in subgraph.outputs:
        tensor = tensors[t_idx]
        g.make_tensor_output(name_prefix + tensor.name, indexed=False, allow_untyped_output=True)


# ---------------------------------------------------------------------------
# Op proxy
# ---------------------------------------------------------------------------


class _OpProxy:
    """Thin wrapper that adapts a :class:`~yobx.litert.litert_helper.TFLiteOperator`
    so converter functions can access inputs/outputs as **string names**
    (consistent with the TF converter style)."""

    __slots__ = "_op", "builtin_options", "custom_code", "inputs", "opcode", "outputs"

    def __init__(
        self, op: TFLiteOperator, input_names: List[str], output_names: List[str]
    ) -> None:
        self._op = op
        self.inputs: Tuple[str, ...] = tuple(input_names)
        self.outputs: Tuple[str, ...] = tuple(output_names)
        self.builtin_options: Dict = op.builtin_options
        self.opcode: int = op.opcode
        self.custom_code: str = op.custom_code

    @property
    def name(self) -> str:
        return self._op.name


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_NP_DTYPE_TO_ONNX: Dict[np.dtype, int] = {
    np.dtype("float32"): onnx.TensorProto.FLOAT,
    np.dtype("float64"): onnx.TensorProto.DOUBLE,
    np.dtype("float16"): onnx.TensorProto.FLOAT16,
    np.dtype("int8"): onnx.TensorProto.INT8,
    np.dtype("int16"): onnx.TensorProto.INT16,
    np.dtype("int32"): onnx.TensorProto.INT32,
    np.dtype("int64"): onnx.TensorProto.INT64,
    np.dtype("uint8"): onnx.TensorProto.UINT8,
    np.dtype("uint16"): onnx.TensorProto.UINT16,
    np.dtype("uint32"): onnx.TensorProto.UINT32,
    np.dtype("uint64"): onnx.TensorProto.UINT64,
    np.dtype("bool"): onnx.TensorProto.BOOL,
}


def _onnx_elem_type_from_np(dtype: np.dtype) -> int:
    return _NP_DTYPE_TO_ONNX.get(dtype, onnx.TensorProto.FLOAT)


_TFLITE_TO_ONNX_DTYPE: Dict[int, int] = {
    0: onnx.TensorProto.FLOAT,  # FLOAT32
    1: onnx.TensorProto.FLOAT16,  # FLOAT16
    2: onnx.TensorProto.INT32,  # INT32
    3: onnx.TensorProto.UINT8,  # UINT8
    4: onnx.TensorProto.INT64,  # INT64
    6: onnx.TensorProto.BOOL,  # BOOL
    7: onnx.TensorProto.INT16,  # INT16
    9: onnx.TensorProto.INT8,  # INT8
    10: onnx.TensorProto.DOUBLE,  # FLOAT64
    12: onnx.TensorProto.UINT64,  # UINT64
    14: onnx.TensorProto.UINT32,  # UINT32
    15: onnx.TensorProto.UINT16,  # UINT16
}


def _tflite_dtype_to_onnx_elem_type(dtype_int: int) -> int:
    return _TFLITE_TO_ONNX_DTYPE.get(dtype_int, onnx.TensorProto.FLOAT)
