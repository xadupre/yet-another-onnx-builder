from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from ..xbuilder import GraphBuilder
from .register import get_tf_op_converter
from .tensorflow_helper import sanitize_name, tf_dtype_to_np_dtype


def to_onnx(
    model,
    args: Tuple[Any],
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Tuple[Dict[int, str]]] = None,
    target_opset: Union[int, Dict[str, int]] = 20,
    verbose: int = 0,
    extra_converters: Optional[Dict[str, Callable]] = None,
):
    """
    Converts a :epkg:`TensorFlow`/:epkg:`Keras` model into ONNX.

    The model is first traced with :func:`get_concrete_function` to obtain the
    actual TensorFlow computation graph.  Each operation in that graph is then
    converted individually to an equivalent ONNX node by a registered converter.

    :param model: a Keras model or layer (must be built / called at least once)
    :param args: dummy inputs (numpy arrays); used to determine dtypes and shapes
    :param input_names: optional list of names for the ONNX input tensors
    :param dynamic_shapes: optional per-input axis-to-dim-name mappings.
        When *None*, axis 0 is treated as a dynamic batch dimension for every input.
    :param target_opset: opset to use; either an integer for the default domain
        (``""``), or a dictionary mapping domain names to opset versions
    :param verbose: verbosity level (0 = silent)
    :param extra_converters: optional mapping from TF op-type string to converter
        function with signature ``(g, sts, outputs, op, verbose=0)``;
        entries here take priority over the built-in op converters
    :return: onnx model
    """
    from . import register_tensorflow_converters

    register_tensorflow_converters()

    if input_names is not None and len(input_names) != len(args):
        raise ValueError(f"Length mismatch: {len(args)=} but input_names={input_names!r}")
    if input_names is None:
        input_names = ["X"] if len(args) == 1 else [f"X{i}" for i in range(len(args))]

    # Build TensorSpec objects for tracing (make batch dim dynamic by default).
    input_specs = _build_input_specs(input_names, args, dynamic_shapes)

    # Trace the model to obtain a concrete TF computation graph.
    if hasattr(model, "get_concrete_function"):
        cf = model.get_concrete_function(*input_specs)
    else:
        fn = tf.function(model)
        cf = fn.get_concrete_function(*input_specs)

    # Populate an ONNX GraphBuilder by walking the concrete-function graph.
    g = GraphBuilder(target_opset)
    _convert_concrete_function(cf, g, args, input_specs, verbose, extra_converters or {})
    return g.to_onnx()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_input_specs(input_names, args, dynamic_shapes):
    """Builds a :class:`tensorflow.TensorSpec` for each dummy input."""
    specs = []
    for i, (name, arg) in enumerate(zip(input_names, args)):
        arr = np.asarray(arg)
        shape = list(arr.shape)
        if dynamic_shapes and i < len(dynamic_shapes):
            for axis, _ in dynamic_shapes[i].items():
                shape[axis] = None
        elif shape:
            shape[0] = None  # default: make the batch dimension dynamic
        specs.append(tf.TensorSpec(shape=shape, dtype=tf.as_dtype(arr.dtype), name=name))
    return specs


def _shape_to_tuple(g: GraphBuilder, shape: tf.TensorShape) -> Tuple[Union[int, str], ...]:
    return tuple(dim if dim is not None else g.unique_dimension_name("dim") for dim in shape)


def _convert_concrete_function(
    cf,
    g: GraphBuilder,
    args,
    input_specs,
    verbose: int,
    extra_converters: Dict[str, Callable],
) -> None:
    """Walks the concrete-function graph and emits equivalent ONNX operations.

    The conversion context ``sts`` maps every TF tensor name to the sanitized
    ONNX tensor name used inside the :class:`~yobx.xbuilder.GraphBuilder`.
    TF tensor names contain characters that are invalid or problematic in ONNX
    (e.g. ``":"`` and ``"/"`` in names like ``"dense/MatMul:0"``); sanitization
    is applied via :func:`~yobx.tensorflow.tensorflow_helper.sanitize_name`
    before registering any tensor with the builder.

    :param cf: a :class:`tensorflow.ConcreteFunction`
    :param g: the :class:`~yobx.xbuilder.GraphBuilder` to populate
    :param args: original dummy inputs (used for dtype / shape information)
    :param input_specs: TensorSpec objects used for tracing
    :param verbose: verbosity level
    :param extra_converters: op-type → converter overrides
    """
    # sts: maps every TF tensor name → its sanitized ONNX tensor name.
    # Op converters receive this dict so they can translate input references.
    sts: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Seed the context with captured variable values.
    #    cf.captured_inputs — list of TF tensors (resource handles) captured
    #    from outside the function (one per trainable variable).
    #    cf.variables        — corresponding tf.Variable objects.
    # ------------------------------------------------------------------
    initializer_values: Dict[str, np.ndarray] = {}
    for _captured_tensor, var in zip(cf.captured_inputs, cf.variables):
        value = var.numpy()
        initializer_values[var.name] = value
        onnx_name = sanitize_name(var.name)
        g.make_initializer(onnx_name, value, source="_convert_concrete_function.0")
        sts[var.name] = onnx_name

    handle_names: Dict[str, str] = {}
    for capture, var in cf.graph.captures:
        handle_names[var.name] = capture._name
        # Propagate the ONNX name to the internal capture handle tensor as well.
        if var.name in sts:
            sts[capture._name] = sts[var.name]

    # ------------------------------------------------------------------
    # 2. Register ONNX inputs for each non-captured Placeholder op.
    # 3. Convert each operation in topological (graph-definition) order.
    # ------------------------------------------------------------------
    set_input_names = {f"{i.name}:0": (ind, i) for ind, i in enumerate(input_specs)}
    for op in cf.graph.get_operations():
        if op.type == "Placeholder":
            tensor = op.outputs[0]
            name = tensor.name
            if name in set_input_names:
                spec = set_input_names[name][1]
                onnx_name = sanitize_name(name)
                g.make_tensor_input(
                    onnx_name, tf_dtype_to_np_dtype(spec.dtype), _shape_to_tuple(g, spec.shape)
                )
                sts[name] = onnx_name
                continue
            if name in initializer_values:
                # Captured variable resource handle — register its numpy value
                # as an ONNX initializer so downstream ReadVariableOp can use it.
                if name not in sts:
                    onnx_name = sanitize_name(name)
                    assert not g.has_name(
                        onnx_name
                    ), f"The name {onnx_name!r} is already taken.{g.get_debug_msg()}"
                    g.make_initializer(
                        onnx_name, initializer_values[name], source="_convert_concrete_function.0"
                    )
                    sts[name] = onnx_name
                continue
            if name in handle_names:
                source_tf_name = handle_names[name]
                # source_tf_name should already be in sts from step 1,
                # where sts[capture._name] = sts[var.name] was set.
                assert source_tf_name in sts, (
                    f"Capture handle {source_tf_name!r} not found in name map; "
                    f"known names: {sorted(sts)}{g.get_debug_msg()}"
                )
                source_onnx_name = sts[source_tf_name]
                onnx_name = sanitize_name(name)
                g.op.Identity(
                    source_onnx_name,
                    outputs=[onnx_name],
                    name="initializer",
                    source="_convert_concrete_function.1",
                )
                sts[name] = onnx_name
                continue
            raise AssertionError(
                f"name={name!r} could not be handled as a placeholder, {type(tensor)=}, "
                f"initializer_values={sorted(initializer_values)}, "
                f"handle_names={sorted(handle_names)}, "
                f"input_names={sorted(set_input_names)}{g.get_debug_msg()}"
            )

        op_type = op.type
        # extra_converters take priority over built-in ones.
        fct = extra_converters.get(op_type) or get_tf_op_converter(op_type)
        if fct is None:
            raise RuntimeError(f"Type {op_type!r} has no converting function mapped to it.")

        op_outputs = [t.name for t in op.outputs]
        onnx_outputs = [sanitize_name(n) for n in op_outputs]
        for tf_n, onnx_n in zip(op_outputs, onnx_outputs):
            sts[tf_n] = onnx_n
        fct(g, sts, onnx_outputs, op)
        assert all(g.has_name(onnx_n) for onnx_n in onnx_outputs), (
            f"Issue with node {op.type}({[i.name for i in op.inputs]}) -> "
            f"{[o.name for o in op.outputs]} ({fct=}){g.get_debug_msg()}"
        )

    # ------------------------------------------------------------------
    # 4. Register ONNX outputs.
    # ------------------------------------------------------------------
    for tensor in cf.outputs:
        assert tensor.name in sts, (
            f"Output tensor {tensor.name!r} not found in name map after converting all ops; "
            f"known names: {sorted(sts)}{g.get_debug_msg()}"
        )
        g.make_tensor_output(sts[tensor.name], indexed=False, allow_untyped_output=True)
