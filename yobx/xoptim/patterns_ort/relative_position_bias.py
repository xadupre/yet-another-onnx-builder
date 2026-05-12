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


class GatedRelativePositionBiasPattern(PatternOptimization):
    """
    Implements the fusion of gated relative position bias computation (DeBERTa-v2/v3 style)
    into ``com.microsoft.GatedRelativePositionBias``.

    The fused pattern corresponds to the DeBERTa disentangled self-attention
    gating computation, which applies a learned sigmoid gate to modulate a
    pre-computed relative position bias tensor.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_query_layer(["query_layer FLOAT(batch, seq_len, num_heads*head_size)"])
            I_rel_pos(["rel_pos FLOAT(1, num_heads, seq_len, seq_len)"])
            i_query_bias["query_bias FLOAT(num_heads*head_size)"]
            i_gate_weight["gate_weight FLOAT(head_size, D)"]
            i_gate_bias["gate_bias FLOAT(D)"]
            i_eco_a["eco_a FLOAT(1, num_heads, 1, 1)"]

            Add_0[["Add(query_layer, query_bias)"]]
            Reshape_1[["Reshape(., [batch, seq_len, num_heads, head_size])"]]
            Transpose_2[["Transpose(., perm=[0,2,1,3])"]]
            MatMul_3[["MatMul(., gate_weight)"]]
            Add_4[["Add(., gate_bias)"]]
            Reshape_5[["Reshape(., [batch, num_heads, seq_len, 2, D//2])"]]
            ReduceSum_6[["ReduceSum(., axis=-1, keepdims=0)"]]
            Sigmoid_7[["Sigmoid(.)"]]
            Split_8[["Split(., axis=-1)"]]
            Mul_9[["Mul(gate_r, eco_a)"]]
            Sub_10[["Sub(., 1.0)"]]
            Mul_11[["Mul(gate_u, .)"]]
            Add_12[["Add(., 2.0)"]]
            Mul_13[["Mul(gate_u_1, rel_pos)"]]

            I_query_layer -->|"FLOAT(batch, seq_len, num_heads*head_size)"| Add_0
            i_query_bias -->|"FLOAT(num_heads*head_size)"| Add_0
            Add_0 --> Reshape_1
            Reshape_1 --> Transpose_2
            Transpose_2 --> MatMul_3
            i_gate_weight -->|"FLOAT(head_size, D)"| MatMul_3
            MatMul_3 --> Add_4
            i_gate_bias -->|"FLOAT(D)"| Add_4
            Add_4 --> Reshape_5
            Reshape_5 --> ReduceSum_6
            ReduceSum_6 --> Sigmoid_7
            Sigmoid_7 --> Split_8
            Split_8 -->|"gate_r"| Mul_9
            i_eco_a -->|"FLOAT(1, num_heads, 1, 1)"| Mul_9
            Mul_9 --> Sub_10
            Split_8 -->|"gate_u"| Mul_11
            Sub_10 --> Mul_11
            Mul_11 --> Add_12
            Add_12 --> Mul_13
            I_rel_pos -->|"FLOAT(1, num_heads, seq_len, seq_len)"| Mul_13

            O_Y(["Y FLOAT(batch, num_heads, seq_len, seq_len)"])
            Mul_13 --> O_Y

            class I_query_layer,I_rel_pos,O_Y ioNode
            class i_query_bias,i_gate_weight,i_gate_bias,i_eco_a initNode
            class Add_0,Reshape_1,Transpose_2,MatMul_3,Add_4 opNode
            class Reshape_5,ReduceSum_6,Sigmoid_7,Split_8 opNode
            class Mul_9,Sub_10,Mul_11,Add_12,Mul_13 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_query_layer(["query_layer FLOAT(batch, seq_len, num_heads*head_size)"])
            I_rel_pos(["rel_pos FLOAT(1, num_heads, seq_len, seq_len)"])
            i_query_bias["query_bias FLOAT(num_heads*head_size)"]
            i_gate_weight["gate_weight FLOAT(head_size, D)"]
            i_gate_bias["gate_bias FLOAT(D)"]
            i_eco_a["eco_a FLOAT(1, num_heads, 1, 1)"]

            GatedRPB_0[["com.microsoft.GatedRelativePositionBias(., ., ., ., ., .)"]]

            I_query_layer -->|"FLOAT(batch, seq_len, num_heads*head_size)"| GatedRPB_0
            i_query_bias -->|"FLOAT(num_heads*head_size)"| GatedRPB_0
            I_rel_pos -->|"FLOAT(1, num_heads, seq_len, seq_len)"| GatedRPB_0
            i_gate_weight -->|"FLOAT(head_size, D)"| GatedRPB_0
            i_gate_bias -->|"FLOAT(D)"| GatedRPB_0
            i_eco_a -->|"FLOAT(1, num_heads, 1, 1)"| GatedRPB_0

            O_Y(["Y FLOAT(batch, num_heads, seq_len, seq_len)"])
            GatedRPB_0 --> O_Y

            class I_query_layer,I_rel_pos,O_Y ioNode
            class i_query_bias,i_gate_weight,i_gate_bias,i_eco_a initNode
            class GatedRPB_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        # The final operation is Mul(gate_u_1, rel_pos).
        if node.op_type != "Mul" or node.domain != "":
            return self.none()

        # Identify gate_u_1 = Add(Mul(gate_u, Sub(Mul(gate_r, eco_a), 1.0)), 2.0).
        # Either input[0] or input[1] could be gate_u_1.
        add_two_node = None
        for gate_u_1_idx in range(2):
            candidate = g.node_before(node.input[gate_u_1_idx])
            if candidate is not None and candidate.op_type == "Add" and candidate.domain == "":
                add_two_node = candidate
                break
        if add_two_node is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Add must have a constant 2.0 input.
        const_two_idx = None
        for i in range(2):
            if g.is_constant_scalar(add_two_node.input[i]):
                if abs(g.get_constant_scalar(add_two_node.input[i]) - 2.0) < 1e-5:
                    const_two_idx = i
                    break
        if const_two_idx is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Add.input[1-const_two_idx] = Mul(gate_u, sub_result).
        mul_gate_u_node = g.node_before(add_two_node.input[1 - const_two_idx])
        if (
            mul_gate_u_node is None
            or mul_gate_u_node.op_type != "Mul"
            or mul_gate_u_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # Identify sub_result = Sub(Mul(gate_r, eco_a), 1.0) among the inputs of mul_gate_u_node.
        sub_node = None
        gate_u_out = None
        for i in range(2):
            candidate_sub = g.node_before(mul_gate_u_node.input[i])
            if (
                candidate_sub is not None
                and candidate_sub.op_type == "Sub"
                and candidate_sub.domain == ""
            ):
                if (
                    g.is_constant_scalar(candidate_sub.input[1])
                    and abs(g.get_constant_scalar(candidate_sub.input[1]) - 1.0) < 1e-5
                ):
                    sub_node = candidate_sub
                    gate_u_out = mul_gate_u_node.input[1 - i]
                    break
        if sub_node is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Sub.input[0] = Mul(gate_r, eco_a).
        mul_eco_a_node = g.node_before(sub_node.input[0])
        if (
            mul_eco_a_node is None
            or mul_eco_a_node.op_type != "Mul"
            or mul_eco_a_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # eco_a must be a constant (shape (1, num_heads, 1, 1)).
        eco_a = None
        gate_r_out = None
        for i in range(2):
            if g.is_constant(mul_eco_a_node.input[i]):
                eco_a = mul_eco_a_node.input[i]
                gate_r_out = mul_eco_a_node.input[1 - i]
                break
        if eco_a is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # gate_u and gate_r must come from the same Split node.
        split_node = g.node_before(gate_u_out)
        if split_node is None or split_node.op_type != "Split" or split_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if len(split_node.output) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if gate_r_out not in split_node.output:
            return self.none(node, inspect.currentframe().f_lineno)

        # Split must be along the last axis (-1 or 3 for a 4-D tensor).
        split_axis = g.get_attributes_with_default(split_node, axis=0)["axis"]
        if split_axis not in (-1, 3):
            return self.none(node, inspect.currentframe().f_lineno)

        # Sigmoid -> ReduceSum -> Reshape -> Add(gate_bias) -> MatMul(query_t, gate_weight).
        sigmoid_node = g.node_before(split_node.input[0])
        if sigmoid_node is None or sigmoid_node.op_type != "Sigmoid" or sigmoid_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        reduce_sum_node = g.node_before(sigmoid_node.input[0])
        if (
            reduce_sum_node is None
            or reduce_sum_node.op_type != "ReduceSum"
            or reduce_sum_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        reshape_2_node = g.node_before(reduce_sum_node.input[0])
        if (
            reshape_2_node is None
            or reshape_2_node.op_type != "Reshape"
            or reshape_2_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        add_gate_bias_node = g.node_before(reshape_2_node.input[0])
        if (
            add_gate_bias_node is None
            or add_gate_bias_node.op_type != "Add"
            or add_gate_bias_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # Identify MatMul and gate_bias from the Add node.
        matmul_node = None
        for i in range(2):
            candidate_mm = g.node_before(add_gate_bias_node.input[i])
            if (
                candidate_mm is not None
                and candidate_mm.op_type == "MatMul"
                and candidate_mm.domain == ""
                and g.is_constant(candidate_mm.input[1])
            ):
                candidate_bias = add_gate_bias_node.input[1 - i]
                if g.is_constant(candidate_bias):
                    matmul_node = candidate_mm
                    break
        if matmul_node is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # MatMul.input[0] = Transpose.
        transpose_node = g.node_before(matmul_node.input[0])
        if (
            transpose_node is None
            or transpose_node.op_type != "Transpose"
            or transpose_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # Transpose.input[0] = Reshape.
        reshape_1_node = g.node_before(transpose_node.input[0])
        if (
            reshape_1_node is None
            or reshape_1_node.op_type != "Reshape"
            or reshape_1_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # Reshape.input[0] = Add(query_layer, query_bias).
        add_query_bias_node = g.node_before(reshape_1_node.input[0])
        if (
            add_query_bias_node is None
            or add_query_bias_node.op_type != "Add"
            or add_query_bias_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # One input is query_layer (dynamic), the other is query_bias (constant).
        query_bias = None
        for i in range(2):
            if g.is_constant(add_query_bias_node.input[i]):
                query_bias = add_query_bias_node.input[i]
                break
        if query_bias is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # eco_a must be a 4-D tensor of shape (1, num_heads, 1, 1).
        eco_a_arr = g.get_computed_constant(eco_a)
        if eco_a_arr is None or eco_a_arr.ndim != 4:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [
                add_query_bias_node,
                reshape_1_node,
                transpose_node,
                matmul_node,
                add_gate_bias_node,
                reshape_2_node,
                reduce_sum_node,
                sigmoid_node,
                split_node,
                mul_eco_a_node,
                sub_node,
                mul_gate_u_node,
                add_two_node,
                node,  # final_mul_node
            ],
            self.apply,
            insert_at=node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        add_query_bias_node: NodeProto,
        reshape_1_node: NodeProto,
        transpose_node: NodeProto,
        matmul_node: NodeProto,
        add_gate_bias_node: NodeProto,
        reshape_2_node: NodeProto,
        reduce_sum_node: NodeProto,
        sigmoid_node: NodeProto,
        split_node: NodeProto,
        mul_eco_a_node: NodeProto,
        sub_node: NodeProto,
        mul_gate_u_node: NodeProto,
        add_two_node: NodeProto,
        final_mul_node: NodeProto,
    ) -> List[NodeProto]:
        # Extract query_layer and query_bias from Add(query_layer, query_bias).
        if g.is_constant(add_query_bias_node.input[0]):
            query_bias = add_query_bias_node.input[0]
            query_layer = add_query_bias_node.input[1]
        else:
            query_bias = add_query_bias_node.input[1]
            query_layer = add_query_bias_node.input[0]

        # Gate projection weight and bias.
        gate_weight = matmul_node.input[1]
        matmul_out = matmul_node.output[0]
        if add_gate_bias_node.input[0] == matmul_out:
            gate_bias = add_gate_bias_node.input[1]
        else:
            gate_bias = add_gate_bias_node.input[0]

        # eco_a from Mul(gate_r, eco_a).
        if g.is_constant(mul_eco_a_node.input[0]):
            eco_a = mul_eco_a_node.input[0]
        else:
            eco_a = mul_eco_a_node.input[1]

        # rel_pos is the input of the final Mul that is NOT gate_u_1.
        gate_u_1_out = add_two_node.output[0]
        if final_mul_node.input[0] == gate_u_1_out:
            rel_pos = final_mul_node.input[1]
        else:
            rel_pos = final_mul_node.input[0]

        # Determine num_heads from eco_a shape (1, num_heads, 1, 1).
        eco_a_arr = g.get_computed_constant(eco_a)
        num_heads = int(eco_a_arr.shape[1])

        grpb_node = g.make_node(
            "GatedRelativePositionBias",
            [query_layer, query_bias, rel_pos, gate_weight, gate_bias, eco_a],
            final_mul_node.output,
            domain="com.microsoft",
            num_heads=num_heads,
            name=f"{self.__class__.__name__}--{final_mul_node.name}",
        )

        return [grpb_node]
