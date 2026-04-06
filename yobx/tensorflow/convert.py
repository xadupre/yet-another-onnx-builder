from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import ValueInfoProto
import tensorflow as tf
from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
from ..helpers.onnx_helper import tensor_dtype_to_np_dtype
from ..xbuilder import GraphBuilder, OptimizationOptions
from .register import get_tf_op_converter
from .tensorflow_helper import tf_dtype_to_np_dtype


def to_onnx(
    model,
    args: Tuple[Any],
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Tuple[Dict[int, str]]] = None,
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    builder_cls: Union[type, Callable] = GraphBuilder,
    verbose: int = 0,
    extra_converters: Optional[Dict[str, Callable]] = None,
    large_model: bool = False,
    external_threshold: int = 1024,
    filename: Optional[str] = None,
    return_optimize_report: bool = False,
) -> ExportArtifact:
    """
    Converts a :epkg:`TensorFlow`/:epkg:`Keras` model into ONNX.

    The model is first traced with :func:`get_concrete_function` to obtain the
    actual TensorFlow computation graph.  Each operation in that graph is then
    converted individually to an equivalent ONNX node by a registered converter.

    :param model: a Keras model or layer (must be built / called at least once)
    :param args: dummy inputs (numpy arrays **or** :class:`onnx.ValueInfoProto`
        descriptors); used to determine dtypes and shapes.  When a
        :class:`~onnx.ValueInfoProto` is provided no actual data is needed —
        the element type and shape are taken directly from the proto.
    :param input_names: optional list of names for the ONNX input tensors
    :param dynamic_shapes: optional per-input axis-to-dim-name mappings.
        When *None*, axis 0 is treated as a dynamic batch dimension for every input.
    :param target_opset: opset to use; either an integer for the default domain
        (``""``), or a dictionary mapping domain names to opset versions.
        If it includes ``{'com.microsoft': 1}``, the converted model
        may include optimized kernels specific to :epkg:`onnxruntime`.
    :param builder_cls: by default the graph builder is a
        :class:`yobx.xbuilder.GraphBuilder` but any builder can
        be used as long it implements the apis :ref:`builder-api`
        and :ref:`builder-api-make`
    :param verbose: verbosity level (0 = silent)
    :param extra_converters: optional mapping from TF op-type string to converter
        function with signature ``(g, sts, outputs, op, verbose=0)``;
        entries here take priority over the built-in op converters
    :param large_model: if True the returned :class:`~yobx.container.ExportArtifact`
        has its :attr:`~yobx.container.ExportArtifact.container` attribute set to
        an :class:`~yobx.container.ExtendedModelContainer`
    :param external_threshold: if ``large_model`` is True, every tensor whose
        element count exceeds this threshold is stored as external data
    :param filename: if set, the exported ONNX model is saved to this path and
        the :class:`~yobx.container.ExportReport` is written as a companion
        Excel file (same base name with ``.xlsx`` extension).
    :param return_optimize_report: if True, the returned
        :class:`~yobx.container.ExportArtifact` has its
        :attr:`~yobx.container.ExportArtifact.report` attribute populated with
        per-pattern optimization statistics
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX proto together with an :class:`~yobx.container.ExportReport`.

    Example::

        import numpy as np
        import tensorflow as tf
        from yobx.tensorflow import to_onnx

        model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
        X = np.random.randn(5, 3).astype(np.float32)

        artifact = to_onnx(model, (X,))
        proto = artifact.proto
        artifact.save("model.onnx")
    """
    from . import register_tensorflow_converters

    if isinstance(target_opset, int):
        dict_target_opset = {"": target_opset}
    else:
        if not isinstance(target_opset, dict):
            raise TypeError(f"target_opset must be a dictionary or an integer not {target_opset}")
        dict_target_opset = target_opset.copy()
        if "" not in dict_target_opset:
            dict_target_opset[""] = 21

    register_tensorflow_converters()

    if input_names is not None and len(input_names) != len(args):
        raise ValueError(f"Length mismatch: {len(args)=} but input_names={input_names!r}")
    if input_names is None:
        # For ValueInfoProto arguments use the embedded name as the default.
        default_names = []
        for j, arg in enumerate(args):
            if isinstance(arg, ValueInfoProto):
                default_names.append(arg.name or (f"X{j}" if len(args) > 1 else "X"))
            else:
                default_names.append("X" if len(args) == 1 else f"X{j}")
        input_names = default_names

    # Build TensorSpec objects for tracing (make batch dim dynamic by default).
    input_specs = _build_input_specs(input_names, args, dynamic_shapes)

    # Trace the model to obtain a concrete TF computation graph.
    if isinstance(model, tf.types.experimental.ConcreteFunction):
        # Model is already a ConcreteFunction (e.g. from jax_to_concrete_function).
        # Use it directly and take the input specs from its signature so the
        # placeholder tensor names in the graph match the lookup in
        # _convert_concrete_function.
        cf = model
        input_specs = list(cf.structured_input_signature[0])
    elif hasattr(model, "get_concrete_function"):
        cf = model.get_concrete_function(*input_specs)
    else:
        # For plain Python callables: try the standard tf.function path first.
        # JAX functions fail here because they cannot accept TF symbolic tensors
        # (the tracer raises a TypeError mentioning an "abstract array" or
        # "SymbolicTensor"). When that happens and JAX is installed, fall back
        # to jax_to_concrete_function; otherwise, surface a clear ImportError.
        try:
            cf = tf.function(model).get_concrete_function(*input_specs)
        except TypeError as e:
            msg = str(e)
            # Only treat the error as JAX-related if it matches the known
            # JAX tracing failures.
            if "abstract array" in msg or "SymbolicTensor" in msg:
                try:
                    from .tensorflow_helper import jax_to_concrete_function
                except ImportError as import_error:
                    raise ImportError(
                        "Converting JAX-based models to ONNX requires 'jax' and "
                        "'jax2tf' to be installed. Please install these "
                        "dependencies and try again."
                    ) from import_error

                cf = jax_to_concrete_function(
                    model, args, input_names=input_names, dynamic_shapes=dynamic_shapes
                )
                # Update input_specs to match the concrete function's actual
                # input signature (jax2tf may rename inputs internally).
                input_specs = list(cf.structured_input_signature[0])
            else:
                # Re-raise non-JAX-related TypeErrors so they are not masked.
                raise

    # Populate an ONNX GraphBuilder by walking the concrete-function graph.
    kwargs = (
        dict(optimization_options=OptimizationOptions(patterns="default+onnxruntime"))
        if "com.microsoft" in dict_target_opset
        else {}
    )

    if verbose:
        if issubclass(builder_cls, GraphBuilder):  # type: ignore
            kwargs["verbose"] = verbose  # type: ignore

        from ..helpers import string_type

        print(f"[yobx.tensorflow.to_onnx] export with builder {builder_cls!r})")
        print(f"[yobx.tensorflow.to_onnx] {dict_target_opset=}")
        print(f"[yobx.tensorflow.to_onnx] args={string_type(args, with_shape=True)}")
        if extra_converters:
            print(f"[yobx.tensorflow.to_onnx] with {len(extra_converters)} extra converters")

    g = builder_cls(dict_target_opset, **kwargs)  # type: ignore

    _convert_concrete_function(cf, g, args, input_specs, verbose, extra_converters or {})
    if isinstance(g, GraphBuilder):
        onx = g.to_onnx(  # type: ignore
            large_model=large_model,
            external_threshold=external_threshold,
            return_optimize_report=return_optimize_report,
        )
        if verbose and onx.report and onx.report.stats:
            import pandas

            print(f"[yobx.tensorflow.to_onnx] done, output type is {type(onx)}")

            df = pandas.DataFrame(onx.report.stats)
            for c in ["added", "removed"]:
                df[c] = df[c].fillna(0).astype(int)
            agg = df.groupby("pattern")[["added", "removed", "time_in"]].sum()
            agg = agg[(agg["added"] > 0) | (agg["removed"] > 0)].sort_values(
                "removed", ascending=False
            )
            if agg.shape[0]:
                print(agg.to_string())
        if filename:
            if verbose:
                print(f"[yobx.tensorflow.to_onnx] saving model to {filename!r}")
            onx.save(filename)
        return onx
    onx = g.to_onnx(large_model=large_model, external_threshold=external_threshold)
    if filename:
        if verbose:
            print(f"[yobx.tensorflow.to_onnx] saving model to {filename!r}")
        onx.save(filename)
    return onx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_input_specs(input_names, args, dynamic_shapes):
    """Builds a :class:`tensorflow.TensorSpec` for each dummy input.

    Each element of *args* may be a numpy array **or** an
    :class:`onnx.ValueInfoProto`.  In the latter case the shape and dtype are
    extracted from the proto; ``None`` is used for any dimension that is not
    a positive integer (symbolic or unknown dimensions are treated as dynamic).
    """
    specs = []
    for i, (name, arg) in enumerate(zip(input_names, args)):
        if isinstance(arg, ValueInfoProto):
            # Extract dtype and shape from the ValueInfoProto.
            tt = arg.type.tensor_type
            np_dtype = tensor_dtype_to_np_dtype(tt.elem_type)
            if tt.HasField("shape"):
                shape = [
                    (None if dim.dim_param else (dim.dim_value or None))
                    for idim, dim in enumerate(tt.shape.dim)
                ]
            else:
                shape = [None]
            specs.append(tf.TensorSpec(shape=shape, dtype=tf.as_dtype(np_dtype), name=name))
        else:
            arr = np.asarray(arg)
            shape = list(arr.shape)
            if dynamic_shapes and i < len(dynamic_shapes):
                for axis, _ in dynamic_shapes[i].items():
                    shape[axis] = None  # type: ignore
            elif shape:
                shape[0] = None  # type: ignore
            specs.append(tf.TensorSpec(shape=shape, dtype=tf.as_dtype(arr.dtype), name=name))
    return specs


def _shape_to_tuple(g: GraphBuilder, shape: tf.TensorShape) -> Tuple[Union[int, str], ...]:
    return tuple(dim if dim is not None else g.unique_dimension_name("dim") for dim in shape)


def _convert_concrete_function(
    cf, g: GraphBuilder, args, input_specs, verbose: int, extra_converters: Dict[str, Callable]
) -> None:
    """
    Walks the concrete-function graph and emits equivalent ONNX operations.

    :param cf: a :class:`tensorflow.ConcreteFunction`
    :param g: the :class:`~yobx.xbuilder.GraphBuilder` to populate
    :param args: original dummy inputs (used for dtype / shape information)
    :param input_specs: TensorSpec objects used for tracing
    :param verbose: verbosity level
    :param extra_converters: op-type → converter overrides
    """
    # ------------------------------------------------------------------
    # 1. Seed the context with captured variable values.
    #    cf.captured_inputs — list of TF tensors (resource handles) captured
    #    from outside the function (one per trainable variable).
    #    cf.variables        — corresponding tf.Variable objects.
    # ------------------------------------------------------------------
    # using keras or not keras, tf does not seem consistent into
    # what it does in both case.
    ops = cf.graph.get_operations()
    initializer_values: Dict[str, Any] = {}
    value_alias = {}
    forbidden = set()
    # Build a mapping from resource handle unique_id to variable for correct pairing.
    # Using zip(cf.captured_inputs, cf.variables) is unreliable: cf.captured_inputs can
    # include non-variable captures (e.g. training-phase flags) and the ordering of the
    # two sequences is not guaranteed to match for multi-layer Keras models.  Instead we
    # match each captured tensor to its variable by comparing handle unique IDs.
    uid_to_var = {var.handle._unique_id: var for var in cf.variables}
    for captured_tensor in cf.captured_inputs:
        var = uid_to_var.get(captured_tensor._unique_id)
        if var is None:
            # Non-variable capture (e.g. a training-phase boolean); skip.
            continue
        # Keep the TF variable as-is; numpy conversion is deferred to export time.
        value = var.value()
        name = f"{var.name}[{captured_tensor._unique_id}]"
        initializer_values[name] = value
        assert not g.has_name(name), f"name {name!r} is already taken."
        g.make_initializer(name, value, source="_convert_concrete_function.0")
        assert tuple(value.shape) == tuple(var.shape) or tuple(value.shape) != tuple(
            captured_tensor.shape
        ), (
            f"Shape Mismatch for {var.name!r}, {var.shape=}, {tuple(value.shape)=}, "
            f"{captured_tensor.shape=}"
        )

        assert (
            captured_tensor._unique_id not in value_alias
        ), f"A unique id {captured_tensor._unique_id!r} is not unique for var={var.name!r}"
        value_alias[captured_tensor._unique_id] = name
        if var.name in forbidden:
            continue
        if var.name in value_alias:
            # In that case, it means there are local variable receiving the same name.
            del value_alias[var.name]
            forbidden.add(var.name)
        else:
            value_alias[var.name] = name

    handle_names: Dict[str, Dict[str, str]] = {}
    for capture, var in cf.graph.captures:
        name = var.name
        assert name not in handle_names, f"Duplicated name={name!r} in {handle_names=}."
        handle_names[name] = dict(
            full_name=f"{capture._name}[{capture._unique_id}]",
            name=capture._name,
            unique_id=capture._unique_id,
        )

    # ------------------------------------------------------------------
    # 2. Register ONNX inputs for each non-captured Placeholder op.
    # 3. Convert each operation in topological (graph-definition) order.
    # ------------------------------------------------------------------
    set_input_names = {f"{i.name}:0": (ind, i) for ind, i in enumerate(input_specs)}
    for op in ops:
        if op.type == "Placeholder":
            tensor = op.outputs[0]
            name = tensor.name
            if name in set_input_names:
                assert not g.has_name(name), f"Input {name!r} is already used{g.get_debug_msg()}"
                spec = set_input_names[name][1]
                g.make_tensor_input(
                    name, tf_dtype_to_np_dtype(spec.dtype), _shape_to_tuple(g, spec.shape)
                )
                continue

            if name in initializer_values:
                # Captured variable resource handle — register its numpy value
                # as an ONNX initializer so downstream ReadVariableOp can use it.
                assert not g.has_name(
                    name
                ), f"Initializer {name!r} is already used{g.get_debug_msg()}"
                g.make_initializer(
                    name, initializer_values[name], source="_convert_concrete_function.0"
                )
                continue
            if name in handle_names:
                assert not g.has_name(
                    name
                ), f"Initializer {name!r} is already used{g.get_debug_msg()}"
                original_name = handle_names[name]["full_name"]
                if not g.has_name(original_name):
                    # We need to add the initializer to the model.
                    if original_name not in initializer_values:
                        unique_id = handle_names[name]["unique_id"]
                        if unique_id in value_alias:
                            original_name = value_alias[unique_id]

                if not g.has_name(original_name):
                    assert original_name in initializer_values, (
                        f"{original_name!r} not found in "
                        f"initializer_values={sorted(initializer_values)}, "
                        f"tensor.name={tensor.name!r}, handle_names={handle_names[name]}, "
                        f"value_alias={value_alias}"
                    )
                    g.make_initializer(
                        original_name,
                        initializer_values[original_name],
                        source="_convert_concrete_function.1",
                    )
                g.op.Identity(original_name, outputs=[name], name="handle")
                continue

            raise AssertionError(
                f"tensor.name={tensor.name!r} could not be handled as a placeholder, "
                f"{type(tensor)=}, initializer_values={sorted(initializer_values)}, "
                f"handle_names={handle_names}, value_alias={value_alias}, "
                f"input_names={sorted(set_input_names)}{g.get_debug_msg()}"
            )

        op_type = op.type
        # extra_converters take priority over built-in ones.
        fct = extra_converters.get(op_type) or get_tf_op_converter(op_type)
        if fct is None:
            raise RuntimeError(
                f"Type {op_type!r} has no converting function mapped to it, "
                f"inputs={[t.name for t in op.inputs]}, "
                f"outputs={[t.name for t in op.outputs]}"
                f"{g.get_debug_msg()}"
            )

        assert all(g.has_shape(i.name) for i in op.inputs), (
            f"A shape is missing, input names={[i.name for i in op.inputs]}, "
            f"has_shape={[g.has_shape(i.name) for i in op.inputs]}"
        )
        onnx_outputs = op_outputs = [t.name for t in op.outputs]
        sts: Dict[str, Any] = {}
        fct(g, sts, onnx_outputs, op)
        assert all(g.has_name(o) for o in op_outputs), (
            f"Issue with node {op.type}({[i.name for i in op.inputs]}) -> "
            f"{[o.name for o in op.outputs]} ({fct=}){g.get_debug_msg()}"
            f"{g.get_debug_msg()}"
        )
        assert all(g.has_shape(i.name) for i in op.outputs), (
            f"A shape is missing, output names={[i.name for i in op.outputs]}, "
            f"has_shape={[g.has_shape(i.name) for i in op.outputs]}{g.get_debug_msg()}"
        )

    # ------------------------------------------------------------------
    # 4. Register ONNX outputs.
    # ------------------------------------------------------------------
    for tensor in cf.outputs:
        g.make_tensor_output(tensor.name, indexed=False, allow_untyped_output=True)
