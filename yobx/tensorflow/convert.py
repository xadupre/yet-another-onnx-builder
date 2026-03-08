from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
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
    initializer_values: Dict[str, np.ndarray] = {}
    value_alias = {}
    forbidden = set()
    for _captured_tensor, var in zip(cf.captured_inputs, cf.variables):
        value = var.numpy()
        name = f"{var.name}[{_captured_tensor._unique_id}]"
        initializer_values[name] = value
        assert not g.has_name(name), f"name {name!r} is already taken."
        g.make_initializer(name, value, source="_convert_concrete_function.0")
        assert value.shape == var.shape or value.shape != _captured_tensor.shape, (
            f"Shape Mismatch for {var.name!r}, {var.shape=}, {value.shape=}, "
            f"{_captured_tensor.shape=}"
        )

        assert (
            _captured_tensor._unique_id not in value_alias
        ), f"A unique id {_captured_tensor._unique_id!r} is not unique for var={var.name!r}"
        value_alias[_captured_tensor._unique_id] = name
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
