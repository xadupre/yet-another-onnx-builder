from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder
from .register import get_tf_op_converter
from .tensorflow_helper import sanitize_name


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
    import tensorflow as tf

    from . import register_tensorflow_converters

    register_tensorflow_converters()

    if input_names is not None and len(input_names) != len(args):
        raise ValueError(f"Length mismatch: {len(args)=} but input_names={input_names!r}")
    if input_names is None:
        input_names = ["X"] if len(args) == 1 else [f"X{i}" for i in range(len(args))]

    # Build TensorSpec objects for tracing (make batch dim dynamic by default).
    input_specs = _build_input_specs(args, dynamic_shapes)

    # Trace the model to obtain a concrete TF computation graph.
    if hasattr(model, "get_concrete_function"):
        cf = model.get_concrete_function(*input_specs)
    else:
        fn = tf.function(model)
        cf = fn.get_concrete_function(*input_specs)

    # Populate an ONNX GraphBuilder by walking the concrete-function graph.
    g = GraphBuilder(target_opset)
    _convert_concrete_function(cf, g, args, input_names, verbose, extra_converters or {})
    return g.to_onnx()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_input_specs(args, dynamic_shapes):
    """Builds a :class:`tensorflow.TensorSpec` for each dummy input."""
    import tensorflow as tf

    specs = []
    for i, arg in enumerate(args):
        arr = np.asarray(arg)
        shape = list(arr.shape)
        if dynamic_shapes and i < len(dynamic_shapes):
            for axis, _ in dynamic_shapes[i].items():
                shape[axis] = None
        elif shape:
            shape[0] = None  # default: make the batch dimension dynamic
        specs.append(tf.TensorSpec(shape=shape, dtype=tf.as_dtype(arr.dtype)))
    return specs


def _convert_concrete_function(
    cf,
    g: GraphBuilder,
    args,
    input_names,
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
    #    cf.captured_inputs — list of TF tensors (resource handles or values)
    #    captured from outside the function (one per trainable variable).
    #    cf.variables        — corresponding tf.Variable objects.
    # ------------------------------------------------------------------
    ctx: Dict[str, Any] = {}
    captured_names: set = set()
    for tensor, var in zip(cf.captured_inputs, cf.variables):
        ctx[tensor.name] = var.numpy()
        captured_names.add(tensor.name)

    # ------------------------------------------------------------------
    # 2. Register ONNX inputs for each non-captured Placeholder op.
    # ------------------------------------------------------------------
    input_idx = 0
    for op in cf.graph.get_operations():
        if op.type != "Placeholder":
            continue
        tensor = op.outputs[0]
        if tensor.name in captured_names:
            continue  # captured variable handle — already in ctx as numpy array
        if input_idx >= len(input_names):
            break
        name = input_names[input_idx]
        arr = np.asarray(args[input_idx])
        # Use the traced shape (may contain None for dynamic dims).
        if tensor.shape.rank is not None:
            onnx_shape = tuple(
                f"dim_{j}" if (dim is None) else dim
                for j, dim in enumerate(tensor.shape.as_list())
            )
        else:
            onnx_shape = tuple(arr.shape)
        g.make_tensor_input(name, np_dtype_to_tensor_dtype(arr.dtype), onnx_shape, device=-1)
        ctx[tensor.name] = name
        input_idx += 1

    # ------------------------------------------------------------------
    # 3. Convert each operation in topological (graph-definition) order.
    # ------------------------------------------------------------------
    for op in cf.graph.get_operations():
        if op.type == "Placeholder":
            continue  # already handled above

        op_type = op.type
        # extra_converters take priority over built-in ones.
        fct = extra_converters.get(op_type) or get_tf_op_converter(op_type)
        if fct is None:
            if verbose:
                print(
                    f"[tensorflow.to_onnx] skipping unsupported op: "
                    f"{op_type} ({op.name!r})"
                )
            continue

        op_outputs = [sanitize_name(t.name) for t in op.outputs]
        fct(g, ctx, op_outputs, op, verbose=verbose)

    # ------------------------------------------------------------------
    # 4. Register ONNX outputs.
    # ------------------------------------------------------------------
    for tensor in cf.outputs:
        onnx_name = ctx.get(tensor.name)
        if isinstance(onnx_name, str):
            g.make_tensor_output(onnx_name, indexed=False, allow_untyped_output=True)
