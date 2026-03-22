from typing import Dict, List, NoReturn, Optional, Sequence, Tuple, Union
import onnx
from onnx import FunctionProto, GraphProto, NodeProto, TensorProto

_i1_o1_node_types = {
    "Abs",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Ceil",
    "Celu",
    "Clip",
    "Cos",
    "Cosh",
    "CumProd",
    "CumSum",
    "DepthToSpace",
    "Elu",
    "Erf",
    "Exp",
    "Expand",
    "FastGelu",
    "Flatten",
    "Floor",
    "Gather",
    "GatherElements",
    "GatherND",
    "Gelu",
    "GlobalMaxPool",
    "HardSigmoid",
    "HardSwish",
    "Identity",
    "LeakyRelu",
    "Log",
    "LogSoftmax",
    "LpNormalization",
    "LRN",
    "MeanVarianceNormalization",
    "Mish",
    "Neg",
    "PRelu",
    "Reciprocal",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
    "Relu",
    "Reshape",
    "RotaryEmbedding",
    "Round",
    "ScatterND",
    "Selu",
    "Shrink",
    "Sign",
    "Sigmoid",
    "Sin",
    "Sinh",
    "Slice",
    "Softmax",
    "Softplus",
    "Softsign",
    "SpaceToDepth",
    "Sqrt",
    "Squeeze",
    "Swish",
    "Tan",
    "Tanh",
    "ThresholdedRelu",
    "Tile",
    "Transpose",
    "TreeEnsemble",
    "Trilu",
    "Trunc",
    "Unsqueeze",
    "Upsample",
}

_in_o1_node_types = {
    "Add",
    "BitShift",
    "BitwiseAnd",
    "BitwiseNot",
    "BitwiseOr",
    "BitwiseXor",
    "Concat",
    "Div",
    "Einsum",
    "FusedMatMul",
    "Gemm",
    "MatMul",
    "Max",
    "Mean",
    "Min",
    "Mod",
    "Mul",
    "Sub",
    "Sum",
    "Unsqueeze",
}


def infer_types(
    node: Union[FunctionProto, NodeProto],
    input_types: Sequence[int],
    output_name: Optional[str] = None,
    exc: bool = True,
) -> Union[int, Sequence[int]]:
    """
    Tries to infer the type of an output or all outputs.

    :param node: NodeProto
    :param input_types: type of the elements of the input tensors
    :param output_name: type for the desired output or
        all types if all are needed
    :param exc: raise an exception if type cannot be inferred
    :return: tuple of types or output type
    """
    if isinstance(node, FunctionProto):
        assert (
            output_name is None
        ), f"output_name must be None if proto is a FunctionProto but output_name={output_name!r}"
        return _infer_types_function(node, input_types, exc=exc)
    return _infer_types_node(node, input_types, output_name, exc=exc)


def _infer_types_function(
    proto: FunctionProto, input_types: Sequence[int], exc: bool = True
) -> Tuple[int, ...]:
    """
    Tries to infer the type of an output or all outputs.

    :param proto: FunctionProto
    :param input_types: type of the elements of the input tensors
    :param output_name: type for the desired output or
        all types if all are needed
    :param exc: raise an exception if type cannot be inferred
    :return: tuple of types or output type
    """
    current = dict(zip(proto.input, input_types))
    for node in proto.node:
        out = _infer_types_node(node, [current[i] for i in node.input], None, exc=exc)
        current.update(dict(zip(node.output, out)))  # type: ignore[arg-type]
    return tuple(current[n] for n in proto.output)


def _infer_types_node(
    node: NodeProto, input_types: Sequence[int], output_name: Optional[str], exc: bool = True
) -> Union[int, Sequence[int]]:
    """
    Tries to infer the type of an output or all outputs.

    :param node: NodeProto
    :param input_types: type of the elements of the input tensors
    :param output_name: type for the desired output or
        all types if all are needed
    :param exc: raise an exception if type cannot be inferred
    :return: tuple of types or output type
    """
    all_types: Optional[Sequence[int]] = None
    if node.op_type in _i1_o1_node_types:
        all_types = _infer_type_i1_o1(node, input_types)
    elif node.op_type in _in_o1_node_types:
        all_types = _infer_type_in_o1(node, input_types)
    elif node.op_type in _dict_type_inference:
        all_types = _dict_type_inference[node.op_type](node, input_types)
    else:
        all_types = None

    if not all_types:
        if exc:
            raise RuntimeError(
                f"Unable to infer type for node type {node.op_type!r}, node is {node}."
            )
        return tuple(0 for _ in node.output) if len(node.output) > 1 else 0

    if output_name:
        assert (
            len(node.output) == 1
        ), f"Unexpected number of outputs {node.output} for node type {node.op_type!r}"
        assert (
            node.output[0] == output_name
        ), f"Output {output_name!r} not in node.output {node.output}"
        return all_types[0]
    return all_types


def _raise_exc(node: NodeProto, input_types: Sequence[int]) -> NoReturn:
    raise RuntimeError(
        f"Unable to guess output type for node type {node.op_type!r}, "
        f"input_types={input_types}, node={node}"
    )


def _infer_type_i1_o1(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """The node has one output and its type is the same as the first input type."""
    assert (
        len(node.output) == 1
    ), f"Unexpected number of outputs {node.output} for node type {node.op_type!r}"
    assert len(input_types) >= 1, (
        f"Unexpected number of inputs {len(input_types)} "
        f"for node type {node.op_type!r}, node is {node}"
    )
    return (input_types[0],)


def _infer_type_in_o1(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """The node has one output and its type is the same as all inputs."""
    assert (
        len(node.output) == 1
    ), f"Unexpected number of outputs {node.output} for node type {node.op_type!r}"
    assert len(input_types) >= 1, (
        f"Unexpected number of inputs {len(input_types)} "
        f"for node type {node.op_type!r}, node is {node}"
    )
    dist = set(i for i in input_types if i != 0)
    if not dist:
        return (0,)
    assert len(dist) == 1, (
        f"Type mismatch for node type {node.op_type!r}, "
        f"input_types={input_types} in node {node}"
    )
    return (max(input_types),)


def _infer_type_cast(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Cast."""
    for att in node.attribute:
        if att.name == "to":
            return (att.i,)
    _raise_exc(node, input_types)


def _infer_type_constant(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Constant."""
    for att in node.attribute:
        if att.name in ("value_int", "value_ints"):
            return (TensorProto.INT64,)
        if att.name in ("value_float", "value_floats"):
            return (TensorProto.FLOAT,)
        if att.name in ("value",):
            return (att.t.data_type,)
    _raise_exc(node, input_types)


def _infer_type_cast_like(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node CastLike."""
    assert len(input_types) == 2, f"Missing input types {input_types}"
    return (input_types[1],)


def _infer_type_constant_of_shape(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Cast."""
    if len(node.attribute) == 0:
        return (TensorProto.FLOAT,)
    value = node.attribute[0]
    return (value.t.data_type,)


def _infer_type_eye_like(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node CastLike."""
    for att in node.attribute:
        if att.name == "dtype":
            return (att.i,)
    return (input_types[0],)


def _infer_type_pow(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a Pow node (same type as first input)."""
    return (input_types[0],)


def _infer_type_range(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Cast."""
    if len(node.input) == 3:
        # starts, ends, axis
        return (max(input_types[:2]),)
    _raise_exc(node, input_types)


def _infer_type_shape_or_size(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Shape."""
    return (TensorProto.INT64,)


def _infer_type_split(node: NodeProto, input_types: Sequence[int]) -> List[int]:
    """Returns the output type for a node Split."""
    return [input_types[0] for _ in node.output]


def _infer_type_where(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Where."""
    return (max(input_types[1:]),)


def _infer_type_dropout(node: NodeProto, input_types: Sequence[int]) -> List[int]:
    """Returns the output types for a node Dropout."""
    types = [input_types[0]]
    if len(node.output) > 1 and node.output[1]:
        types.append(TensorProto.BOOL)
    return types


def _infer_type_bool_output_unary(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns BOOL for unary bool-output ops (Not, IsInf, IsNaN)."""
    return (TensorProto.BOOL,)


def _infer_type_bool_output_binary(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns BOOL for comparison/logical ops."""
    return (TensorProto.BOOL,)


def _infer_type_arg_max_min(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns INT64 for ArgMax/ArgMin."""
    return (TensorProto.INT64,)


def _infer_type_non_zero(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns INT64 for NonZero."""
    return (TensorProto.INT64,)


def _infer_type_window(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns output type for window functions (BlackmanWindow, HammingWindow, HannWindow)."""
    for att in node.attribute:
        if att.name == "output_datatype":
            return (att.i,)
    return (TensorProto.FLOAT,)


def _infer_type_topk(node: NodeProto, input_types: Sequence[int]) -> Tuple[int, int]:
    """Returns (same type as input, INT64) for TopK."""
    return (input_types[0], TensorProto.INT64)


def _infer_type_batch_normalization(node: NodeProto, input_types: Sequence[int]) -> List[int]:
    """Returns output types for BatchNormalization."""
    types = [input_types[0]]
    for _ in range(len(node.output) - 1):
        types.append(input_types[0])
    return types


def _infer_type_onehot(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the type of the values input (input[2]) for OneHot."""
    if len(input_types) >= 3 and input_types[2]:
        return (input_types[2],)
    return (TensorProto.FLOAT,)


def _infer_type_mel_weight_matrix(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns output type for MelWeightMatrix."""
    for att in node.attribute:
        if att.name == "output_datatype":
            return (att.i,)
    return (TensorProto.FLOAT,)


def _infer_type_qlinear_matmul(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns output type for QLinearMatMul (type of output scale = input[5])."""
    if len(input_types) >= 6 and input_types[5]:
        return (input_types[5],)
    return (TensorProto.FLOAT,)


def _infer_type_rnn(node: NodeProto, input_types: Sequence[int]) -> List[int]:
    """Returns output types for RNN/GRU/LSTM (same type as input data)."""
    itype = input_types[0] if input_types else TensorProto.FLOAT
    return [itype for _ in node.output]


def _infer_types_body(
    body: GraphProto, input_types: Sequence[int], exc: bool = False
) -> Dict[str, int]:
    """
    Propagates element types through a body subgraph (GraphProto).

    :param body: body subgraph
    :param input_types: element types for the body graph inputs (in order)
    :param exc: raise an exception if a node type cannot be inferred
    :return: mapping from result name to element type
    """
    current: Dict[str, int] = {}
    # Initializer types
    for init in body.initializer:
        current[init.name] = init.data_type
    # Body input types: prefer explicitly declared type; fall back to input_types
    for i, inp in enumerate(body.input):
        try:
            declared = inp.type.tensor_type.elem_type if inp.type.HasField("tensor_type") else 0
        except AttributeError:
            declared = 0
        itype = declared or (input_types[i] if i < len(input_types) else 0)
        if itype:
            current[inp.name] = itype
    # Propagate through nodes
    for node in body.node:
        node_input_types = [current.get(n, 0) for n in node.input]
        out = _infer_types_node(node, node_input_types, None, exc=exc)
        if isinstance(out, int):
            out = [out]
        for name, t in zip(node.output, out):
            if name and t:
                current[name] = t
    return current


def _infer_type_loop(node: NodeProto, input_types: Sequence[int]) -> List[int]:
    """Returns output types for Loop inferred from the body subgraph outputs.

    The body subgraph outputs are ordered as:
    [cond_out, v_out_0, ..., v_out_K-1, scan_0, ..., scan_S-1]

    The Loop node outputs are:
    [v_final_0, ..., v_final_K-1, scan_0, ..., scan_S-1]

    So Loop output[i] has the same element type as body output[i+1].

    When body output element types are not explicitly declared (0), type
    inference is propagated through the body graph nodes.  Loop body inputs
    are mapped as follows:
    - body.input[0] (iteration counter) → INT64
    - body.input[1] (condition)         → BOOL
    - body.input[2+] (loop-carried vars) → input_types[2+]
    """
    for att in node.attribute:
        if att.type == onnx.AttributeProto.GRAPH:
            body = att.g
            # body.output[0] is cond_out (BOOL), skip it
            # body.output[1..] match loop outputs one-to-one
            body_out_types = [
                o.type.tensor_type.elem_type if o.type.HasField("tensor_type") else 0
                for o in body.output
            ]
            if len(body_out_types) > 1:
                # If any type is still unknown (0), propagate types through the
                # body graph to fill them in.
                if any(t == 0 for t in body_out_types[1:]):
                    body_input_types = [
                        TensorProto.INT64,  # iter
                        TensorProto.BOOL,  # cond_in
                    ] + list(input_types[2:])
                    inferred = _infer_types_body(body, body_input_types)
                    body_out_types = [
                        inferred.get(
                            o.name,
                            o.type.tensor_type.elem_type if o.type.HasField("tensor_type") else 0,
                        )
                        for o in body.output
                    ]
                return body_out_types[1:]
            break
    return [0 for _ in node.output]


_dict_type_inference = {
    "And": _infer_type_bool_output_binary,
    "ArgMax": _infer_type_arg_max_min,
    "ArgMin": _infer_type_arg_max_min,
    "BatchNormalization": _infer_type_batch_normalization,
    "BitCast": _infer_type_cast,
    "BlackmanWindow": _infer_type_window,
    "Cast": _infer_type_cast,
    "CastLike": _infer_type_cast_like,
    "Constant": _infer_type_constant,
    "ConstantOfShape": _infer_type_constant_of_shape,
    "Dropout": _infer_type_dropout,
    "Equal": _infer_type_bool_output_binary,
    "EyeLike": _infer_type_eye_like,
    "Greater": _infer_type_bool_output_binary,
    "GreaterOrEqual": _infer_type_bool_output_binary,
    "HammingWindow": _infer_type_window,
    "HannWindow": _infer_type_window,
    "IsInf": _infer_type_bool_output_unary,
    "IsNaN": _infer_type_bool_output_unary,
    "Less": _infer_type_bool_output_binary,
    "LessOrEqual": _infer_type_bool_output_binary,
    "Loop": _infer_type_loop,
    "MelWeightMatrix": _infer_type_mel_weight_matrix,
    "NonMaxSuppression": _infer_type_non_zero,
    "NonZero": _infer_type_non_zero,
    "Not": _infer_type_bool_output_unary,
    "OneHot": _infer_type_onehot,
    "Or": _infer_type_bool_output_binary,
    "Pow": _infer_type_pow,
    "QLinearMatMul": _infer_type_qlinear_matmul,
    "Range": _infer_type_range,
    "RNN": _infer_type_rnn,
    "Shape": _infer_type_shape_or_size,
    "Size": _infer_type_shape_or_size,
    "Split": _infer_type_split,
    "TopK": _infer_type_topk,
    "Where": _infer_type_where,
    "Xor": _infer_type_bool_output_binary,
}
