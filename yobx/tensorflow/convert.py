from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder
from .register import get_tf_op_converter
from .tensorflow_helper import tf_dtype_to_np_dtype


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

    The conversion context ``ctx`` maps every TF tensor name to either an ONNX
    tensor name (``str``) for dynamic values, or a :class:`numpy.ndarray` for
    constant / weight values.

    :param cf: a :class:`tensorflow.ConcreteFunction`
    :param g: the :class:`~yobx.xbuilder.GraphBuilder` to populate
    :param args: original dummy inputs (used for dtype / shape information)
    :param input_names: ONNX names for the model inputs
    :param verbose: verbosity level
    :param extra_converters: op-type → converter overrides
    """
    # ------------------------------------------------------------------
    # 1. Seed the context with captured variable values.
    #    cf.captured_inputs — list of TF tensors (resource handles) captured
    #    from outside the function (one per trainable variable).
    #    cf.variables        — corresponding tf.Variable objects.
    #
    #    Key by the captured tensor's name, which matches the Placeholder
    #    output tensor name seen in cf.graph.get_operations().
    # ------------------------------------------------------------------
    initializer_values: Dict[str, np.ndarray] = {}
    for captured_tensor, var in zip(cf.captured_inputs, cf.variables):
        initializer_values[captured_tensor.name] = var.numpy()

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
                g.make_tensor_input(
                    name, tf_dtype_to_np_dtype(spec.dtype), _shape_to_tuple(g, spec.shape)
                )
                continue
            if name in initializer_values:
                # Captured variable resource handle — register its numpy value
                # as an ONNX initializer so downstream ReadVariableOp can use it.
                g.make_initializer(
                    name, initializer_values[name], source="_convert_concrete_function"
                )
                continue

        op_type = op.type
        # extra_converters take priority over built-in ones.
        fct = extra_converters.get(op_type) or get_tf_op_converter(op_type)
        if fct is None:
            raise RuntimeError(f"Type {op_type!r} has no converting function mapped to it.")

        op_outputs = [t.name for t in op.outputs]
        fct(g, {}, op_outputs, op)
        assert all(
            g.has_name(o) for o in op_outputs
        ), f"Issue with node {op.type}({[i.name for i in op.inputs]}) -> {[o.name for o in op.outputs]} ({fct=}){g.get_debug_msg()}"

    # ------------------------------------------------------------------
    # 4. Register ONNX outputs.
    # ------------------------------------------------------------------
    for tensor in cf.outputs:
        g.make_tensor_output(tensor.name, indexed=False, allow_untyped_output=True)
