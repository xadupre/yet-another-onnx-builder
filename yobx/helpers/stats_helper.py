"""
Functions to compute statistics on an ONNX model such as number of nodes
per op_type and estimation of computational cost.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import onnx


def _get_attribute_value(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    """Returns the value of an attribute, or *default* if missing."""
    for att in node.attribute:
        if att.name != name:
            continue
        if att.type == onnx.AttributeProto.INT:
            return att.i
        if att.type == onnx.AttributeProto.INTS:
            return list(att.ints)
        if att.type == onnx.AttributeProto.FLOAT:
            return att.f
        if att.type == onnx.AttributeProto.FLOATS:
            return list(att.floats)
        if att.type == onnx.AttributeProto.STRING:
            return att.s
    return default


def _static_size(shape: Optional[Tuple]) -> Optional[int]:
    """Returns the number of elements in a static shape, or None if dynamic."""
    if shape is None:
        return None
    n = 1
    for d in shape:
        if not isinstance(d, int) or d < 0:
            return None
        n *= d
    return n


def _estimate_node_flops(
    node: onnx.NodeProto,
    shape_map: Dict[str, Optional[Tuple]],
) -> Optional[int]:
    """
    Estimates the number of floating-point operations for a single ONNX node.

    Returns None when the shapes are not fully known (dynamic shapes) or the
    op_type is not covered.

    :param node: ONNX node
    :param shape_map: mapping from tensor name to shape tuple
    :return: estimated number of FLOPs, or None
    """
    op = node.op_type

    def shape(name: str) -> Optional[Tuple]:
        return shape_map.get(name)

    def size(name: str) -> Optional[int]:
        return _static_size(shape(name))

    # ---- element-wise unary (1 op / element) ----
    if op in {
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BitShift",
        "Ceil",
        "Celu",
        "Cos",
        "Cosh",
        "Elu",
        "Erf",
        "Exp",
        "Floor",
        "HardSigmoid",
        "HardSwish",
        "LeakyRelu",
        "Log",
        "Mish",
        "Neg",
        "Not",
        "Relu",
        "Round",
        "Selu",
        "Shrink",
        "Sign",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Tan",
        "Tanh",
        "ThresholdedRelu",
    }:
        if node.output:
            return size(node.output[0])
        return None

    # ---- element-wise binary (1 op / element) ----
    if op in {
        "Add",
        "And",
        "BitAnd",
        "BitOr",
        "BitXor",
        "Div",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "Mod",
        "Mul",
        "Or",
        "Pow",
        "PRelu",
        "Sub",
        "Xor",
    }:
        if node.output:
            return size(node.output[0])
        return None

    # ---- Sigmoid (1 exp + 1 add + 1 div ≈ 3 ops / element) ----
    if op == "Sigmoid":
        s = size(node.output[0]) if node.output else None
        return None if s is None else 3 * s

    # ---- Softmax / LogSoftmax (exp + sum + div ≈ 3 ops / element) ----
    if op in {"Softmax", "LogSoftmax"}:
        s = size(node.output[0]) if node.output else None
        return None if s is None else 3 * s

    # ---- MatMul [M,K] @ [K,N] → 2*M*K*N ----
    if op == "MatMul":
        if len(node.input) < 2:
            return None
        a = shape(node.input[0])
        b = shape(node.input[1])
        if a is None or b is None:
            return None
        # batched matmul: last two dims are [M,K] and [K,N]
        if len(a) < 2 or len(b) < 2:
            return None
        M, K = a[-2], a[-1]
        K2, N = b[-2], b[-1]
        if not (isinstance(M, int) and isinstance(K, int) and isinstance(N, int)):
            return None
        batch = 1
        for d in a[:-2]:
            if not isinstance(d, int):
                return None
            batch *= d
        return 2 * batch * M * K * N

    # ---- Gemm (alpha * A @ B + beta * C): 2*M*K*N + M*N ----
    if op == "Gemm":
        if len(node.input) < 2:
            return None
        a = shape(node.input[0])
        b = shape(node.input[1])
        if a is None or b is None or len(a) < 2 or len(b) < 2:
            return None
        trans_a = int(_get_attribute_value(node, "transA", 0))
        trans_b = int(_get_attribute_value(node, "transB", 0))
        M, K = (a[1], a[0]) if trans_a else (a[0], a[1])
        K2, N = (b[1], b[0]) if trans_b else (b[0], b[1])
        if not (isinstance(M, int) and isinstance(K, int) and isinstance(N, int)):
            return None
        flops = 2 * M * K * N
        if len(node.input) >= 3 and node.input[2]:
            # bias addition
            flops += M * N
        return flops

    # ---- Conv ----
    if op in {"Conv", "ConvTranspose"}:
        if not node.output:
            return None
        out_shape = shape(node.output[0])
        if out_shape is None or len(out_shape) < 3:
            return None
        # out_shape: [N, C_out, *spatial_out]
        if node.input:
            in_shape = shape(node.input[0])
        else:
            in_shape = None
        if in_shape is None or len(in_shape) < 2:
            return None
        # weight shape: [C_out, C_in/group, *kernel]
        if len(node.input) >= 2 and node.input[1]:
            w_shape = shape(node.input[1])
        else:
            w_shape = None
        if w_shape is None or len(w_shape) < 3:
            return None
        N = out_shape[0]
        C_out = out_shape[1]
        spatial_out = out_shape[2:]
        kernel = w_shape[2:]
        C_in_per_group = w_shape[1]
        if not all(isinstance(d, int) for d in (N, C_out, C_in_per_group)):
            return None
        if not all(isinstance(d, int) for d in spatial_out):
            return None
        if not all(isinstance(d, int) for d in kernel):
            return None
        spatial_out_size = math.prod(spatial_out) if spatial_out else 1
        kernel_size = math.prod(kernel) if kernel else 1
        # 2 ops per MAC (multiply + accumulate)
        return 2 * N * C_out * C_in_per_group * kernel_size * spatial_out_size

    # ---- Pooling ops ----
    if op in {"MaxPool", "AveragePool"}:
        if not node.output:
            return None
        out_shape = shape(node.output[0])
        if out_shape is None or len(out_shape) < 3:
            return None
        spatial_out = out_shape[2:]
        if not all(isinstance(d, int) for d in spatial_out):
            return None
        spatial_out_size = math.prod(spatial_out) if spatial_out else 1
        N = out_shape[0]
        C = out_shape[1]
        if not isinstance(N, int) or not isinstance(C, int):
            return None
        kernel_shape = _get_attribute_value(node, "kernel_shape", None)
        if kernel_shape is None:
            return None
        kernel_size = math.prod(kernel_shape) if kernel_shape else 1
        return N * C * spatial_out_size * kernel_size

    if op in {"GlobalAveragePool", "GlobalMaxPool"}:
        if not node.input:
            return None
        in_shape = shape(node.input[0])
        if in_shape is None or len(in_shape) < 3:
            return None
        N = in_shape[0]
        C = in_shape[1]
        spatial = in_shape[2:]
        if not all(isinstance(d, int) for d in (N, C, *spatial)):
            return None
        return N * C * math.prod(spatial) if spatial else N * C

    # ---- BatchNormalization (mean, var, normalize ≈ 2 ops / element) ----
    if op == "BatchNormalization":
        if node.output:
            s = size(node.output[0])
            return None if s is None else 2 * s
        return None

    # ---- LayerNormalization (mean, var, sub, div, scale, bias ≈ 6 ops / element) ----
    if op in {"LayerNormalization", "GroupNormalization", "InstanceNormalization"}:
        if node.output:
            s = size(node.output[0])
            return None if s is None else 6 * s
        return None

    # ---- Reduce ops ----
    if op in {
        "ReduceMax",
        "ReduceMin",
        "ReduceMean",
        "ReduceSum",
        "ReduceProd",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceSumSquare",
    }:
        if not node.input:
            return None
        in_shape = shape(node.input[0])
        s = _static_size(in_shape)
        return s

    # ---- LSTM / GRU / RNN ----
    # Rough estimate: 4 MatMuls for LSTM (gates), 3 for GRU
    if op == "LSTM":
        if len(node.input) < 2:
            return None
        x_shape = shape(node.input[0])  # [seq, batch, input_size]
        w_shape = shape(node.input[1])  # [num_dir, 4*hidden, input_size]
        if x_shape is None or w_shape is None:
            return None
        if len(x_shape) < 3 or len(w_shape) < 3:
            return None
        seq, batch, input_size = x_shape[0], x_shape[1], x_shape[2]
        hidden_4 = w_shape[1]
        if not all(isinstance(d, int) for d in (seq, batch, input_size, hidden_4)):
            return None
        hidden = hidden_4 // 4
        # per step: 2 MatMuls (input->gates + hidden->gates)
        return 2 * seq * batch * (input_size + hidden) * hidden_4

    if op == "GRU":
        if len(node.input) < 2:
            return None
        x_shape = shape(node.input[0])
        w_shape = shape(node.input[1])
        if x_shape is None or w_shape is None:
            return None
        if len(x_shape) < 3 or len(w_shape) < 3:
            return None
        seq, batch, input_size = x_shape[0], x_shape[1], x_shape[2]
        hidden_3 = w_shape[1]
        if not all(isinstance(d, int) for d in (seq, batch, input_size, hidden_3)):
            return None
        hidden = hidden_3 // 3
        return 2 * seq * batch * (input_size + hidden) * hidden_3

    if op == "RNN":
        if len(node.input) < 2:
            return None
        x_shape = shape(node.input[0])
        w_shape = shape(node.input[1])
        if x_shape is None or w_shape is None:
            return None
        if len(x_shape) < 3 or len(w_shape) < 3:
            return None
        seq, batch, input_size = x_shape[0], x_shape[1], x_shape[2]
        hidden = w_shape[1]
        if not all(isinstance(d, int) for d in (seq, batch, input_size, hidden)):
            return None
        return 2 * seq * batch * (input_size + hidden) * hidden

    # ---- zero-cost data movement ops ----
    if op in {
        "Cast",
        "CastLike",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Expand",
        "Flatten",
        "Gather",
        "GatherElements",
        "GatherND",
        "Identity",
        "OneHot",
        "Pad",
        "Reshape",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Shape",
        "Slice",
        "Split",
        "Squeeze",
        "Tile",
        "Transpose",
        "Unsqueeze",
    }:
        return 0

    return None


def model_statistics(
    model: onnx.ModelProto,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Computes statistics on an ONNX model.

    :param model: ONNX model
    :param verbose: verbosity level
    :return: dictionary with the following keys:

        - ``"n_nodes"`` – total number of nodes
        - ``"node_count_per_op_type"`` – dict mapping op_type → count
        - ``"total_estimated_flops"`` – total estimated FLOPs (int or None)
        - ``"flops_per_op_type"`` – dict mapping op_type → estimated FLOPs
          (None means the cost could not be estimated for some nodes)
        - ``"node_stats"`` – list of per-node dicts with keys
          ``op_type``, ``name``, ``inputs``, ``outputs``, ``estimated_flops``
    """
    from ..xshape import BasicShapeBuilder

    # Run shape inference to collect shapes for all intermediate tensors.
    bs = BasicShapeBuilder(verbose=verbose)
    try:
        bs.run_model(model)
    except Exception:
        # Shape inference may fail for some models; we continue with partial results.
        pass

    # Build a mapping name → shape tuple (or None for dynamic/unknown).
    shape_map: Dict[str, Optional[Tuple]] = {}
    for name in list(bs._known_shapes.keys()):  # type: ignore[attr-defined]
        try:
            sh = bs.get_shape(name)
            # Convert any non-int symbolic dim to None to mark as dynamic
            if all(isinstance(d, int) for d in sh):
                shape_map[name] = tuple(int(d) for d in sh)
            else:
                shape_map[name] = sh
        except Exception:
            shape_map[name] = None

    # Collect per-node stats
    node_count: Dict[str, int] = {}
    flops_by_op: Dict[str, List[Optional[int]]] = {}
    node_stats: List[Dict[str, Any]] = []

    def _collect_nodes(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            op = node.op_type
            node_count[op] = node_count.get(op, 0) + 1

            f = _estimate_node_flops(node, shape_map)
            if op not in flops_by_op:
                flops_by_op[op] = []
            flops_by_op[op].append(f)

            input_shapes = [(n, shape_map.get(n)) for n in node.input if n]
            output_shapes = [(n, shape_map.get(n)) for n in node.output if n]
            node_stats.append(
                {
                    "op_type": op,
                    "name": node.name,
                    "inputs": input_shapes,
                    "outputs": output_shapes,
                    "estimated_flops": f,
                }
            )
            # Recurse into subgraphs (e.g. inside If / Loop / Scan)
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    _collect_nodes(att.g)
                elif att.type == onnx.AttributeProto.GRAPHS:
                    for g in att.graphs:
                        _collect_nodes(g)

    _collect_nodes(model.graph)

    # Aggregate FLOPs per op_type
    flops_per_op: Dict[str, Optional[int]] = {}
    total: Optional[int] = 0
    for op, values in flops_by_op.items():
        op_total: Optional[int] = 0
        for v in values:
            if v is None:
                op_total = None
                break
            op_total += v  # type: ignore[operator]
        flops_per_op[op] = op_total
        if op_total is None:
            total = None
        elif total is not None:
            total += op_total

    return {
        "n_nodes": sum(node_count.values()),
        "node_count_per_op_type": node_count,
        "total_estimated_flops": total,
        "flops_per_op_type": flops_per_op,
        "node_stats": node_stats,
    }
