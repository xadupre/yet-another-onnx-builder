import inspect
import math
from typing import List, Optional

import numpy as np
from onnx import NodeProto, TensorProto

from ..patterns_api import MatchResult, PatternOptimization


class RelativePositionBiasPattern(PatternOptimization):
    """
    Fuses the relative position bias computation (T5-style, encoder) into
    ``com.microsoft.RelativePositionBias``.

    The fused pattern corresponds to the T5 bidirectional relative attention
    bias computation, recognizable by a ``Gather`` node reading from a
    learnable bias table, whose indices are computed through a bucketing
    function of absolute relative positions.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_seq_len(["seq_len INT64()"])
            i_bias_table["bias_table FLOAT(num_buckets, num_heads)"]
            i_zero["zero INT64()"]
            i_one["one INT64()"]
            i_max_exact_int["max_exact INT64()"]
            i_max_exact_float["max_exact FLOAT()"]
            i_log_max["log_max FLOAT()"]
            i_scale["scale FLOAT()"]
            i_clamp["clamp_val INT64()"]

            Range_0[["Range(zero, seq_len, one)"]]
            Unsqueeze_1[["Unsqueeze(., [0])"]]
            Unsqueeze_2[["Unsqueeze(., [1])"]]
            Sub_3[["Sub(., .)"]]
            Abs_4[["Abs(.)"]]
            CastFloat_5[["Cast(., to=FLOAT)"]]
            Div_6[["Div(., max_exact_float)"]]
            Log_7[["Log(.)"]]
            Div_8[["Div(., log_max)"]]
            Mul_9[["Mul(., scale)"]]
            CastInt_10[["Cast(., to=INT64)"]]
            Add_11[["Add(., max_exact_int)"]]
            Shape_12[["Shape(.)"]]
            ConstantOfShape_13[["ConstantOfShape(., clamp_val)"]]
            Min_14[["Min(., .)"]]
            CastInt2_15[["Cast(., to=INT64)"]]
            Where_16[["Where(., ., .)"]]
            Gather_17[["Gather(bias_table, .)"]]
            Transpose_18[["Transpose(., perm=[2,0,1])"]]
            Unsqueeze_19[["Unsqueeze(., [0])"]]

            I_seq_len -->|"INT64()"| Range_0
            Range_0 -->|"INT64(seq_len)"| Unsqueeze_1
            Range_0 -->|"INT64(seq_len)"| Unsqueeze_2
            Unsqueeze_1 -->|"INT64(1, seq_len)"| Sub_3
            Unsqueeze_2 -->|"INT64(seq_len, 1)"| Sub_3
            Sub_3 -->|"INT64(seq_len, seq_len)"| Abs_4
            Abs_4 -->|"INT64(seq_len, seq_len)"| CastFloat_5
            Abs_4 -->|"INT64(seq_len, seq_len)"| CastInt2_15
            CastFloat_5 -->|"FLOAT(seq_len, seq_len)"| Div_6
            Div_6 -->|"FLOAT(seq_len, seq_len)"| Log_7
            Log_7 -->|"FLOAT(seq_len, seq_len)"| Div_8
            Div_8 -->|"FLOAT(seq_len, seq_len)"| Mul_9
            Mul_9 -->|"FLOAT(seq_len, seq_len)"| CastInt_10
            CastInt_10 -->|"INT64(seq_len, seq_len)"| Add_11
            Add_11 -->|"INT64(seq_len, seq_len)"| Shape_12
            Add_11 -->|"INT64(seq_len, seq_len)"| Min_14
            Shape_12 -->|"INT64(2)"| ConstantOfShape_13
            ConstantOfShape_13 -->|"INT64(seq_len, seq_len)"| Min_14
            CastInt2_15 -->|"INT64(seq_len, seq_len)"| Where_16
            Min_14 -->|"INT64(seq_len, seq_len)"| Where_16
            i_bias_table -->|"FLOAT(num_buckets, num_heads)"| Gather_17
            Where_16 -->|"INT64(seq_len, seq_len)"| Gather_17
            Gather_17 -->|"FLOAT(seq_len, seq_len, num_heads)"| Transpose_18
            Transpose_18 -->|"FLOAT(num_heads, seq_len, seq_len)"| Unsqueeze_19

            O_Y(["Y FLOAT(1, num_heads, seq_len, seq_len)"])
            Unsqueeze_19 --> O_Y

            class I_seq_len,O_Y ioNode
            class i_bias_table,i_zero,i_one,i_max_exact_int,i_max_exact_float initNode
            class i_log_max,i_scale,i_clamp initNode
            class Range_0,Unsqueeze_1,Unsqueeze_2,Sub_3,Abs_4 opNode
            class CastFloat_5,Div_6,Log_7,Div_8,Mul_9,CastInt_10 opNode
            class Add_11,Shape_12,ConstantOfShape_13,Min_14 opNode
            class CastInt2_15,Where_16,Gather_17,Transpose_18,Unsqueeze_19 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_seq_len(["seq_len INT64()"])
            i_bias_table_t["bias_table_T FLOAT(num_heads, num_buckets)"]

            RelativePositionBias_0[["com.microsoft.RelativePositionBias(., ., .)"]]

            i_bias_table_t -->|"FLOAT(num_heads, num_buckets)"| RelativePositionBias_0
            I_seq_len -->|"INT64()"| RelativePositionBias_0
            I_seq_len -->|"INT64()"| RelativePositionBias_0

            O_Y(["Y FLOAT(1, num_heads, seq_len, seq_len)"])
            RelativePositionBias_0 --> O_Y

            class I_seq_len,O_Y ioNode
            class i_bias_table_t initNode
            class RelativePositionBias_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gather" or node.domain != "":
            return self.none()

        # input[0] must be a constant bias table (learnable embedding)
        if not g.is_constant(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        # input[1] = bucket indices, must come from a Where node
        where_node = g.node_before(node.input[1])
        if where_node is None or where_node.op_type != "Where" or where_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # False branch of Where (input[2]) = Min (clamp to num_buckets - 1)
        min_node = g.node_before(where_node.input[2])
        if min_node is None or min_node.op_type != "Min" or min_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Min.input[0] = Add (max_exact + log_bucket_index)
        add_bucket_node = g.node_before(min_node.input[0])
        if (
            add_bucket_node is None
            or add_bucket_node.op_type != "Add"
            or add_bucket_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(add_bucket_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        # Min.input[1] = ConstantOfShape (clamp constant with shape = shape of Add output)
        const_of_shape_node = g.node_before(min_node.input[1])
        if (
            const_of_shape_node is None
            or const_of_shape_node.op_type != "ConstantOfShape"
            or const_of_shape_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # ConstantOfShape.input[0] = Shape(Add.output)
        shape_node = g.node_before(const_of_shape_node.input[0])
        if shape_node is None or shape_node.op_type != "Shape" or shape_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if shape_node.input[0] != add_bucket_node.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        # Add.input[0] = Cast to int64 (integer log bucket)
        cast_int_bucket_node = g.node_before(add_bucket_node.input[0])
        if (
            cast_int_bucket_node is None
            or cast_int_bucket_node.op_type != "Cast"
            or cast_int_bucket_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        to_attr = g.get_attribute(cast_int_bucket_node, "to")
        if to_attr.i != TensorProto.INT64:
            return self.none(node, inspect.currentframe().f_lineno)

        # Cast.input[0] = Mul (scale * log_ratio)
        mul_node = g.node_before(cast_int_bucket_node.input[0])
        if mul_node is None or mul_node.op_type != "Mul" or mul_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Mul.input[0] = Div (log_result / log_max)
        div_log_node = g.node_before(mul_node.input[0])
        if div_log_node is None or div_log_node.op_type != "Div" or div_log_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(div_log_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        # Div.input[0] = Log
        log_node = g.node_before(div_log_node.input[0])
        if log_node is None or log_node.op_type != "Log" or log_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Log.input[0] = Div (pos_float / max_exact_float)
        div_pos_node = g.node_before(log_node.input[0])
        if div_pos_node is None or div_pos_node.op_type != "Div" or div_pos_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(div_pos_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        # Div.input[0] = Cast to float (absolute positions as float)
        cast_float_node = g.node_before(div_pos_node.input[0])
        if (
            cast_float_node is None
            or cast_float_node.op_type != "Cast"
            or cast_float_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        to_float_attr = g.get_attribute(cast_float_node, "to")
        if to_float_attr.i not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE):
            return self.none(node, inspect.currentframe().f_lineno)

        # Cast.input[0] = Abs (absolute relative positions, for bidirectional)
        abs_node = g.node_before(cast_float_node.input[0])
        if abs_node is None or abs_node.op_type != "Abs" or abs_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Abs.input[0] = Sub (relative positions: q_i - k_j)
        sub_node = g.node_before(abs_node.input[0])
        if sub_node is None or sub_node.op_type != "Sub" or sub_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Sub.input[0] = Unsqueeze of Range output (query axis, typically axis=0)
        unsqueeze_q_node = g.node_before(sub_node.input[0])
        if (
            unsqueeze_q_node is None
            or unsqueeze_q_node.op_type != "Unsqueeze"
            or unsqueeze_q_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # Sub.input[1] = Unsqueeze of Range output (key axis, typically axis=1)
        unsqueeze_k_node = g.node_before(sub_node.input[1])
        if (
            unsqueeze_k_node is None
            or unsqueeze_k_node.op_type != "Unsqueeze"
            or unsqueeze_k_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # Both Unsqueeze nodes must operate on the same Range output
        if unsqueeze_q_node.input[0] != unsqueeze_k_node.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        range_node = g.node_before(unsqueeze_q_node.input[0])
        if range_node is None or range_node.op_type != "Range" or range_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Forward: Gather -> Transpose -> Unsqueeze (adds batch dimension)
        gather_nexts = g.next_nodes(node.output[0])
        if len(gather_nexts) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        transpose_node = gather_nexts[0]
        if transpose_node.op_type != "Transpose" or transpose_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        transpose_nexts = g.next_nodes(transpose_node.output[0])
        if len(transpose_nexts) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        unsqueeze_batch_node = transpose_nexts[0]
        if unsqueeze_batch_node.op_type != "Unsqueeze" or unsqueeze_batch_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [
                cast_float_node,
                div_pos_node,
                log_node,
                div_log_node,
                mul_node,
                cast_int_bucket_node,
                add_bucket_node,
                shape_node,
                const_of_shape_node,
                min_node,
                where_node,
                node,  # gather_node
                transpose_node,
                unsqueeze_batch_node,
            ],
            self.apply,
            insert_at=unsqueeze_batch_node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        cast_float_node: NodeProto,
        div_pos_node: NodeProto,
        log_node: NodeProto,
        div_log_node: NodeProto,
        mul_node: NodeProto,
        cast_int_bucket_node: NodeProto,
        add_bucket_node: NodeProto,
        shape_node: NodeProto,
        const_of_shape_node: NodeProto,
        min_node: NodeProto,
        where_node: NodeProto,
        gather_node: NodeProto,
        transpose_node: NodeProto,
        unsqueeze_batch_node: NodeProto,
    ) -> List[NodeProto]:
        # Compute max_exact from Add's constant input (integer value)
        max_exact_arr = g.get_computed_constant(add_bucket_node.input[1])
        max_exact = int(max_exact_arr.flat[0])

        # Compute log_max from Div's constant input
        log_max_arr = g.get_computed_constant(div_log_node.input[1])
        log_max_val = float(log_max_arr.flat[0])

        # max_distance = max_exact * exp(log_max), since log_max = log(max_distance / max_exact)
        max_distance = round(max_exact * math.exp(log_max_val))

        # Transpose the bias table from [num_buckets, num_heads] to [num_heads, num_buckets]
        bias_table = g.get_computed_constant(gather_node.input[0])
        bias_table_t = np.transpose(bias_table)
        new_bias_table_name = g.make_initializer(
            "",
            bias_table_t.astype(bias_table.dtype),
            source=f"{self.__class__.__name__}.bias_table",
        )

        # Retrieve seq_len by tracing back through Abs -> Sub -> Unsqueeze -> Range
        abs_node = g.node_before(cast_float_node.input[0])
        sub_node = g.node_before(abs_node.input[0])
        unsqueeze_q_node = g.node_before(sub_node.input[0])
        range_node = g.node_before(unsqueeze_q_node.input[0])
        # Range(start, limit, delta): limit = input[1] = sequence length
        seq_len = range_node.input[1]

        rpb_node = g.make_node(
            "RelativePositionBias",
            [new_bias_table_name, seq_len, seq_len],
            unsqueeze_batch_node.output,
            domain="com.microsoft",
            max_distance=max_distance,
            is_bidirectional=1,
            name=f"{self.__class__.__name__}--{gather_node.name}",
        )

        return [rpb_node]
