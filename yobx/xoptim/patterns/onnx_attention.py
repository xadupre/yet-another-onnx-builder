import inspect
from typing import List, Optional, Tuple, Union
import numpy as np
from onnx import NodeProto, TensorProto
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ...xbuilder import FunctionOptions, GraphBuilder
from ..patterns_api import MatchResult, PatternOptimization


class FunctionAttentionPattern(PatternOptimization):
    """
    Merges Attention nodes into a local function.
    That includes a version for GroupQueryAttention
    (see second pattern).

    Main Pattern
    ++++++++++++

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_values(["values FLOAT(av, bv, cv, dv)"])
            I_keys(["keys FLOAT(ak, bk, ck, dk)"])
            I_scale_sqrt(["scale_sqrt FLOAT(1)"])
            I_mask(["mask BOOL(am, bm, cm, dm)"])
            I_query(["query FLOAT(aq, bq, cq, dq)"])

            Constant_0[["Constant() -#gt; scale_sqrt"]]
            Mul_1[["Mul(., .)"]]
            Mul_2[["Mul(., .)"]]
            Transpose_3[["Transpose(., perm=[0, 1, 3, 2])"]]
            MatMul_4[["MatMul(., .)"]]
            Where_5[["Where(., [0.0], [-inf])"]]
            Add_6[["Add(., .)"]]
            Softmax_7[["Softmax(., axis=-1)"]]
            IsNaN_8[["IsNaN(.)"]]
            Where_9[["Where(., [0.0], .)"]]
            MatMul_10[["MatMul(., .)"]]

            I_query -->|"FLOAT(aq, bq, cq, dq)"| Mul_1
            Constant_0 -->|"FLOAT(1)"| Mul_1
            I_keys -->|"FLOAT(ak, bk, ck, dk)"| Mul_2
            Constant_0 -->|"FLOAT(1)"| Mul_2
            Mul_2 -->|"FLOAT(ak, bk, ck, dk)"| Transpose_3
            Mul_1 -->|"FLOAT(aq, bq, cq, dq)"| MatMul_4
            Transpose_3 -->|"FLOAT(ak, bk, dk, ck)"| MatMul_4
            I_mask -->|"BOOL(am, bm, cm, dm)"| Where_5
            MatMul_4 -->|"FLOAT(aq^ak, bq^bk, cq, ck)"| Add_6
            Where_5 -->|"FLOAT(am, bm, cm, dm)"| Add_6
            Add_6 -->|"FLOAT(aq^ak^am, bq^bk^bm, cq^cm, ck^dm)"| Softmax_7
            Softmax_7 -->|"FLOAT(aq^ak^am, bq^bk^bm, cq^cm, ck^dm)"| IsNaN_8
            IsNaN_8 -->|"BOOL(aq^ak^am, bq^bk^bm, cq^cm, ck^dm)"| Where_9
            Softmax_7 -->|"FLOAT(aq^ak^am, bq^bk^bm, cq^cm, ck^dm)"| Where_9
            Where_9 -->|"FLOAT(aq^ak^am, bq^bk^bm, cq^cm, ck^dm)"| MatMul_10
            I_values -->|"FLOAT(av, bv, cv, dv)"| MatMul_10

            O_Y(["Y FLOAT(ay, by, cy, dy)"])
            MatMul_10 --> O_Y

            class I_values,I_keys,I_scale_sqrt,I_mask,I_query,O_Y ioNode
            class Constant_0 constNode
            class Mul_1,Mul_2,Transpose_3,MatMul_4,Where_5,Add_6,Softmax_7 opNode
            class IsNaN_8,Where_9,MatMul_10 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_values(["values FLOAT(av, bv, cv, dv)"])
            I_keys(["keys FLOAT(ak, bk, ck, dk)"])
            I_scale_sqrt(["scale_sqrt FLOAT(1)"])
            I_mask(["mask BOOL(am, bm, cm, dm)"])
            I_query(["query FLOAT(aq, bq, cq, dq)"])

            LocalAttention_to1_0[["intermediate.LocalAttention_to1(., ., ., ., .)"]]

            I_query -->|"FLOAT(aq, bq, cq, dq)"| LocalAttention_to1_0
            I_keys -->|"FLOAT(ak, bk, ck, dk)"| LocalAttention_to1_0
            I_values -->|"FLOAT(av, bv, cv, dv)"| LocalAttention_to1_0
            I_mask -->|"BOOL(am, bm, cm, dm)"| LocalAttention_to1_0
            I_scale_sqrt -->|"FLOAT(1)"| LocalAttention_to1_0

            O_Y(["Y FLOAT(ay, by, cy, dy)"])
            LocalAttention_to1_0 --> O_Y

            class I_values,I_keys,I_scale_sqrt,I_mask,I_query,O_Y ioNode
            class LocalAttention_to1_0 opNode

    GroupQueryAttention (GQA)
    +++++++++++++++++++++++++

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_init1_s___RSh1(["init1_s_::RSh1 FLOAT(1)"])
            I_query(["query FLOAT(batch, 8, seq_length, 32)"])
            I_cat_1(["cat_1 FLOAT(batch, 4, past_length+seq_length, 32)"])
            I_cat(["cat FLOAT(batch, 4, past_length+seq_length, 32)"])
            I_to(["to BOOL(seq_length, total_length)"])
            I_init7_s4_0_8__1_32(["init7_s4_0_8_-1_32 INT64(4)"])
            I_init7_s5_1_1_2_1_1(["init7_s5_1_1_2_1_1 INT64(5)"])

            Constant_0[["Constant() -#gt; init1_s_::RSh1"]]
            Constant_1[["Constant() -#gt; init7_s5_1_1_2_1_1"]]
            Constant_2[["Constant() -#gt; init7_s4_0_8_-1_32"]]
            Mul_3[["Mul(., .)"]]
            Unsqueeze_4[["Unsqueeze(., [2])"]]
            Mul_5[["Mul(., [0.4204482])"]]
            Expand_6[["Expand(., .)"]]
            Reshape_7[["Reshape(., .)"]]
            Transpose_8[["Transpose(., perm=[0, 1, 3, 2])"]]
            MatMul_9[["MatMul(., .)"]]
            w10[["Where(., [-inf], .)"]]
            s11[["Softmax(., axis=-1)"]]
            nan12[["IsNaN(.)"]]
            w13[["Where(., 0.0, .)"]]
            Unsqueeze_14[["Unsqueeze(., [2])"]]
            Expand_15[["Expand(., .)"]]
            Reshape_16[["Reshape(., .)"]]
            mm17[["MatMul(., .)"]]

            I_query -->|"FLOAT(batch, 8, seq_length, 32)"| Mul_3
            Constant_0 -->|"FLOAT(1)"| Mul_3
            I_cat -->|"FLOAT(batch, 4, past_length+seq_length, 32)"| Unsqueeze_4
            Unsqueeze_4 -->|"FLOAT(batch, 4, 1, past_length+seq_length, 32)"| Mul_5
            Expand_6 -->|"FLOAT(batch, 4, 1, past_length+seq_length, 32)"| Expand_6
            Constant_1 -->|"INT64(5)"| Expand_6
            Expand_6 -->|"FLOAT(batch, 4, 1, past_length+seq_length, 32)"| Reshape_7
            Constant_2 -->|"INT64(4)"| Reshape_7
            Reshape_7 -->|"FLOAT(batch, 8, 128*(past_length+seq_length)//256, 32)"| Transpose_8
            Mul_3 -->|"FLOAT(batch, 8, seq_length, 32)"| MatMul_9
            Transpose_8 -->|"FLOAT(batch, 8, 32, 128*(past_length+seq_length)//256)"| MatMul_9
            I_to -->|"BOOL(seq_length, total_length)"| w10
            MatMul_9 -->|"FLOAT(batch, 8, seq_length, 128*(past_length+seq_length)//256)"| w10
            w10 -->|"FLOAT(batch, 8, seq_length,
            total_length^128*(past_length+seq_length)//256)"| s11
            s11 -->|"FLOAT(batch, 8, seq_length,
            total_length^128*(past_length+seq_length)//256)"| nan12
            nan12 -->|"BOOL(batch, 8, seq_length,
            total_length^128*(past_length+seq_length)//256)"| w13
            s11 -->|"FLOAT(batch, 8, seq_length,
            total_length^128*(past_length+seq_length)//256)"| w13
            I_cat_1 -->|"FLOAT(batch, 4, past_length+seq_length, 32)"| Unsqueeze_14
            Unsqueeze_14 -->|"FLOAT(batch, 4, 1, past_length+seq_length, 32)"| Expand_15
            Constant_1 -->|"INT64(5)"| Expand_15
            Expand_15 -->|"FLOAT(batch, 4, 2, past_length+seq_length, 32)"| Reshape_16
            Constant_2 -->|"INT64(4)"| Reshape_16
            w13 -->|"FLOAT(batch, 8, seq_length,
            total_length^128*(past_length+seq_length)//256)"| mm17
            Reshape_16 -->|"FLOAT(batch, 8, past_length+seq_length, 32)"| mm17

            O_output_0(["output_0 FLOAT(batch, 8, seq_length, 32)"])
            mm17 --> O_output_0

            class I_init1_s___RSh1,I_query,I_cat_1,I_cat,I_to ioNode
            class I_init7_s4_0_8__1_32,I_init7_s5_1_1_2_1_1,O_output_0 ioNode
            class Constant_0,Constant_1,Constant_2 constNode
            class Mul_3,Unsqueeze_4,Mul_5,Expand_6,Reshape_7,Transpose_8 opNode
            class MatMul_9,w10,s11,nan12,w13,Unsqueeze_14,Expand_15 opNode
            class Reshape_16,mm17 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_query(["query FLOAT(batch, 8, seq_length, 32)"])
            I_cat_1(["cat_1 FLOAT(batch, 4, past_length+seq_length, 32)"])
            I_cat(["cat FLOAT(batch, 4, past_length+seq_length, 32)"])
            I_to(["to BOOL(seq_length, total_length)"])
            I_init7_s4_0_8__1_32(["init7_s4_0_8_-1_32 INT64(4)"])
            I_init7_s5_1_1_2_1_1(["init7_s5_1_1_2_1_1 INT64(5)"])
            I_init1_s___RSh1(["init1_s_::RSh1 FLOAT(1)"])

            LocalAttentionGQASW_to1_0[["intermediate.LocalAttentionGQASW_to1(
            ., ., ., ., ., ., .)"]]

            I_query -->|"FLOAT(batch, 8, seq_length, 32)"| LocalAttentionGQASW_to1_0
            I_cat -->|"FLOAT(batch, 4, past_length+seq_length, 32)"| LocalAttentionGQASW_to1_0
            I_cat_1 -->|"FLOAT(batch, 4, past_length+seq_length, 32)"| LocalAttentionGQASW_to1_0
            I_to -->|"BOOL(seq_length, total_length)"| LocalAttentionGQASW_to1_0
            I_init1_s___RSh1 -->|"FLOAT(1)"| LocalAttentionGQASW_to1_0
            I_init7_s5_1_1_2_1_1 -->|"INT64(5)"| LocalAttentionGQASW_to1_0
            I_init7_s4_0_8__1_32 -->|"INT64(4)"| LocalAttentionGQASW_to1_0

            O_output_0(["output_0 FLOAT(batch, 8, seq_length, 32)"])
            LocalAttentionGQASW_to1_0 --> O_output_0

            class I_query,I_cat_1,I_cat,I_to,I_init7_s4_0_8__1_32 ioNode
            class I_init7_s5_1_1_2_1_1,I_init1_s___RSh1,O_output_0 ioNode
            class LocalAttentionGQASW_to1_0 opNode

    3D Pattern
    ++++++++++

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_values_t(["values_t FLOAT(av, 8, cv, 64)"])
            I_keys(["keys FLOAT(ak, ck, bk*dk)"])
            I_scale_sqrt(["scale_sqrt FLOAT(1)"])
            I_shape0(["shape0 INT64(4)"])
            I_mask(["mask BOOL(am, bm, cm, dm)"])
            I_query(["query FLOAT(aq, cq, bq*dq)"])

            Constant_0[["Constant() -#gt; scale_sqrt"]]
            Constant_1[["Constant() -#gt; shape0"]]
            Mul_2[["Mul(., .)"]]
            Transpose_3[["Transpose(., perm=[0, 2, 1, 3])"]]
            Reshape_4[["Reshape(., .)"]]
            Mul_5[["Mul(., .)"]]
            Reshape_6[["Reshape(., .)"]]
            Transpose_7[["Transpose(., perm=[0, 2, 3, 1])"]]
            MatMul_8[["MatMul(., .)"]]
            Where_9[["Where(., [0.0], [-inf])"]]
            Add_10[["Add(., .)"]]
            s11[["Softmax(., axis=-1)"]]
            nan12[["IsNaN(.)"]]
            w13[["Where(., [0.0], .)"]]
            MatMul_14[["MatMul(., .)"]]

            I_query -->|"FLOAT(aq, cq, bq*dq)"| Mul_2
            Constant_0 -->|"FLOAT(1)"| Mul_2
            Reshape_4 -->|"FLOAT(aq, cq, 8, 64)"| Transpose_3
            Mul_2 -->|"FLOAT(aq, cq, bq*dq)"| Reshape_4
            Constant_1 -->|"INT64(4)"| Reshape_4
            I_keys -->|"FLOAT(ak, ck, bk*dk)"| Mul_5
            Constant_0 -->|"FLOAT(1)"| Mul_5
            Mul_5 -->|"FLOAT(ak, ck, bk*dk)"| Reshape_6
            Constant_1 -->|"INT64(4)"| Reshape_6
            Reshape_6 -->|"FLOAT(ak, ck, 8, 64)"| Transpose_7
            Transpose_3 --> MatMul_8
            Transpose_7 -->|"FLOAT(ak, 8, 64, ck)"| MatMul_8
            I_mask -->|"BOOL(am, bm, cm, dm)"| Where_9
            MatMul_8 --> Add_10
            Where_9 -->|"FLOAT(am, bm, cm, dm)"| Add_10
            Add_10 --> s11
            s11 --> nan12
            nan12 --> w13
            s11 --> w13
            w13 --> MatMul_14
            I_values_t -->|"FLOAT(av, 8, cv, 64)"| MatMul_14

            O_Y(["Y FLOAT(ay, by, cy, dy)"])
            MatMul_14 --> O_Y

            class I_values_t,I_keys,I_scale_sqrt,I_shape0,I_mask,I_query,O_Y ioNode
            class Constant_0,Constant_1 constNode
            class Mul_2,Transpose_3,Reshape_4,Mul_5,Reshape_6,Transpose_7 opNode
            class MatMul_8,Where_9,Add_10,s11,nan12,w13,MatMul_14 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_values_t(["values_t FLOAT(av, 8, cv, 64)"])
            I_keys(["keys FLOAT(ak, ck, bk*dk)"])
            I_scale_sqrt(["scale_sqrt FLOAT(1)"])
            I_shape0(["shape0 INT64(4)"])
            I_mask(["mask BOOL(am, bm, cm, dm)"])
            I_query(["query FLOAT(aq, cq, bq*dq)"])

            Reshape_0[["Reshape(., .)"]]
            Reshape_1[["Reshape(., .)"]]
            Transpose_2[["Transpose(., perm=[0, 2, 1, 3])"]]
            Transpose_3[["Transpose(., perm=[0, 2, 1, 3])"]]
            LocalAttention_to1_4[["intermediate.LocalAttention_to1(., ., ., ., .)"]]

            I_query -->|"FLOAT(aq, cq, bq*dq)"| Reshape_0
            I_shape0 -->|"INT64(4)"| Reshape_0
            I_keys -->|"FLOAT(ak, ck, bk*dk)"| Reshape_1
            I_shape0 -->|"INT64(4)"| Reshape_1
            Reshape_0 --> Transpose_2
            Reshape_1 --> Transpose_3
            Transpose_2 --> LocalAttention_to1_4
            Transpose_3 --> LocalAttention_to1_4
            I_values_t -->|"FLOAT(av, 8, cv, 64)"| LocalAttention_to1_4
            I_mask -->|"BOOL(am, bm, cm, dm)"| LocalAttention_to1_4
            I_scale_sqrt -->|"FLOAT(1)"| LocalAttention_to1_4

            O_Y(["Y FLOAT(ay, by, cy, dy)"])
            LocalAttention_to1_4 --> O_Y

            class I_values_t,I_keys,I_scale_sqrt,I_shape0,I_mask,I_query,O_Y ioNode
            class Reshape_0,Reshape_1,Transpose_2,Transpose_3,LocalAttention_to1_4 opNode
    """

    _operator_name = "LocalAttention"
    _domain_name = "intermediate"

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def _find_index_inf(self, g, where_node):
        for i in (1, 2):
            if g.is_constant_scalar(where_node.input[i]):
                dtype = g.get_computed_constant(where_node.input[i]).dtype
                cst = g.get_constant_scalar(where_node.input[i])
                if np.isinf(cst) or cst == np.finfo(dtype).min:
                    return i
        return None

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Softmax" or node.domain != "" or g.main_opset < 18:
            return self.none()
        axis = g.get_attribute(node, "axis").i
        if axis != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        node_before = g.node_before(node.input[0])
        if not node_before:
            return self.none(node, inspect.currentframe().f_lineno)
        if node_before.op_type == "Add":
            # Add(X, Where(mask, 0, -inf))
            add_node = node_before
            where_node = g.node_before(add_node.input[1])
            if where_node is None or where_node.op_type != "Where":
                return self.none(node, inspect.currentframe().f_lineno)

            if not g.is_constant_scalar(where_node.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant_scalar(where_node.input[2]):
                return self.none(node, inspect.currentframe().f_lineno)

            cst_zero = g.get_constant_scalar(where_node.input[1])
            if cst_zero != 0:
                return self.none(node, inspect.currentframe().f_lineno)
            cst_inf = g.get_constant_scalar(where_node.input[2])
            if not np.isinf(cst_inf):
                return self.none(node, inspect.currentframe().f_lineno)

            mat_qk = g.node_before(add_node.input[0])
            if mat_qk is None or mat_qk.op_type not in ("MatMul", "FusedMatMul"):
                return self.none(node, inspect.currentframe().f_lineno)
        elif node_before.op_type == "Where":
            # Where(mask, -np.inf, X)
            add_node = None
            where_node = node_before
            if not g.is_constant_scalar(where_node.input[1]) and not g.is_constant_scalar(
                where_node.input[2]
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            cst_zero = None
            inf_index = 1 if g.is_constant_scalar(where_node.input[1]) else 2
            dtype = g.get_computed_constant(where_node.input[inf_index]).dtype
            cst_inf = g.get_constant_scalar(where_node.input[inf_index])
            if (not np.isinf(cst_inf) and cst_inf != np.finfo(dtype).min) or cst_inf > 0:
                return self.none(node, inspect.currentframe().f_lineno)
            mat_qk = g.node_before(where_node.input[3 - inf_index])
            if mat_qk is None or mat_qk.op_type not in ("MatMul", "FusedMatMul"):
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            return self.none(node, inspect.currentframe().f_lineno)

        mul1 = g.node_before(mat_qk.input[0])
        if mul1 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if mul1.op_type == "Transpose":
            transpose_mul1 = mul1
            reshape_mul1 = g.node_before(mul1.input[0])
            perm = tuple(g.get_attribute(transpose_mul1, "perm").ints)
            if perm != (0, 2, 1, 3):
                return self.none(node, inspect.currentframe().f_lineno)
            if reshape_mul1 is None:
                return self.none(node, inspect.currentframe().f_lineno)
            mul1 = g.node_before(reshape_mul1.input[0])
        else:
            reshape_mul1 = None
            transpose_mul1 = None
        if mul1 is None or mul1.op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(mul1.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        if mat_qk.op_type == "MatMul":
            transpose = g.node_before(mat_qk.input[1])
            if transpose is None or transpose.op_type != "Transpose":
                return self.none(node, inspect.currentframe().f_lineno)
            perm = g.get_attribute(transpose, "perm").ints
            if transpose_mul1 is None:
                if tuple(perm) != (0, 1, 3, 2):
                    return self.none(node, inspect.currentframe().f_lineno)
                mul2 = g.node_before(transpose.input[0])
                reshape_mul2 = None
            else:
                if tuple(perm) != (0, 2, 3, 1):
                    return self.none(node, inspect.currentframe().f_lineno)
                reshape_mul2 = g.node_before(transpose.input[0])
                if reshape_mul2 is None:
                    return self.none(node, inspect.currentframe().f_lineno)
                mul2 = g.node_before(reshape_mul2.input[0])
                if not g.is_constant(reshape_mul1.input[1]) or not g.is_constant(
                    reshape_mul2.input[1]
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
                shapem1 = g.get_computed_constant(reshape_mul1.input[1])
                shapem2 = g.get_computed_constant(reshape_mul2.input[1])
                if shapem1 is None or shapem2 is None:
                    return self.none(node, inspect.currentframe().f_lineno)
                if shapem1.tolist() != shapem2.tolist():
                    return self.none(node, inspect.currentframe().f_lineno)
        else:
            # FusedMatMul
            transA = g.get_attribute_with_default(mat_qk, "transA", 0)
            transB = g.get_attribute_with_default(mat_qk, "transB", 1)
            if transA != 0 or transB != 1:
                return self.none(node, inspect.currentframe().f_lineno)
            transpose = None
            mul2 = g.node_before(mat_qk.input[1])
            reshape_mul2 = None

        if mul2 is None:
            return self.none(node, inspect.currentframe().f_lineno)

        if mul2.op_type == "Mul":
            # This condition is verified for Attention or MultiHeadAttention.
            gqa_expand = gqa_reshape = gqa_unsqueeze = None
        elif mul2.op_type == "Reshape":
            # This condition is verified by GroupQueryAttention.
            gqa_reshape = mul2
            mul2 = None
            gqa_expand = g.node_before(gqa_reshape.input[0])
            if gqa_expand.op_type != "Expand":
                return self.none(node, inspect.currentframe().f_lineno)
            mul2 = g.node_before(gqa_expand.input[0])
            if mul2.op_type != "Mul":
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_unsqueeze = g.node_before(mul2.input[0])
            if gqa_unsqueeze.op_type != "Unsqueeze":
                return self.none(node, inspect.currentframe().f_lineno)
            #
            if not g.is_constant(gqa_expand.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            exp_shape = g.get_computed_constant(gqa_expand.input[1])
            if tuple(exp_shape[:2]) != (1, 1) or tuple(exp_shape[3:]) != (1, 1):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant(gqa_unsqueeze.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            unsq_shape = g.get_computed_constant(gqa_unsqueeze.input[1])
            if tuple(unsq_shape) != (2,):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant(gqa_reshape.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            resh_shape = g.get_computed_constant(gqa_reshape.input[1])
            if resh_shape.size != 4:
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.has_shape(gqa_unsqueeze.input[0]) or not g.has_shape(gqa_reshape.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            shape1 = g.get_shape_renamed(gqa_unsqueeze.input[0])
            shape2 = g.get_shape_renamed(gqa_reshape.output[0])
            if shape1[0] != shape2[0] or shape1[2] != shape2[2] or shape1[3] != shape2[3]:
                return self.none(
                    node,
                    inspect.currentframe().f_lineno,
                    msg=lambda: f"Shape mismatch {shape1=}, {shape2=}",
                )
        else:
            # No Attention, no MultiHeadAttention, no GroupQueryAttention
            return self.none(node, inspect.currentframe().f_lineno)

        if mul2.input[1] != mul1.input[1]:
            if not g.is_constant_scalar(mul1.input[1]) or not g.is_constant_scalar(mul2.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            cst1 = g.get_constant_scalar(mul1.input[1])
            cst2 = g.get_constant_scalar(mul2.input[1])
            if cst1 != cst2:
                return self.none(node, inspect.currentframe().f_lineno)

        # after softmax
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if {n.op_type for n in next_nodes} != {"Where", "IsNaN"}:
            return self.none(node, inspect.currentframe().f_lineno)
        isnan, where2 = next_nodes[:: (1 if next_nodes[0].op_type == "IsNaN" else -1)]
        if where2.input[0] != isnan.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)
        if where2.input[2] != node.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(where2.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_constant_scalar(where2.input[1])
        if cst != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        mat_qkvs = g.next_nodes(where2.output[0])
        if len(mat_qkvs) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        mat_qkv = mat_qkvs[0]
        if mat_qkv.op_type != "MatMul":
            return self.none(node, inspect.currentframe().f_lineno)

        if gqa_reshape:
            # We need to include the nodes repeating values,
            # the same one which repeated the keys.
            gqa_reshape_v = g.node_before(mat_qkv.input[1])
            if gqa_reshape_v.op_type != "Reshape":
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_expand_v = g.node_before(gqa_reshape_v.input[0])
            if gqa_expand_v.op_type != "Expand":
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_unsqueeze_v = g.node_before(gqa_expand_v.input[0])
            if gqa_unsqueeze_v.op_type != "Unsqueeze":
                return self.none(node, inspect.currentframe().f_lineno)
            #
            if not g.is_constant(gqa_expand.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            exp_shape_v = g.get_computed_constant(gqa_expand_v.input[1])
            if tuple(exp_shape) != tuple(exp_shape_v):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant(gqa_unsqueeze_v.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            unsq_shape_v = g.get_computed_constant(gqa_unsqueeze_v.input[1])
            if tuple(unsq_shape_v) != tuple(unsq_shape):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant(gqa_reshape_v.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            resh_shape_v = g.get_computed_constant(gqa_reshape_v.input[1])
            if tuple(resh_shape_v) != tuple(resh_shape):
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            gqa_expand_v = gqa_reshape_v = gqa_unsqueeze_v = None

        nodes = [
            mul1,
            transpose_mul1,
            reshape_mul1,
            gqa_unsqueeze,
            mul2,
            reshape_mul2,
            gqa_expand,
            gqa_reshape,
            transpose,
            mat_qk,
            where_node,
            add_node,
            node,
            isnan,
            where2,
            gqa_unsqueeze_v,
            gqa_expand_v,
            gqa_reshape_v,
            mat_qkv,
        ]

        for n in nodes[:-1]:
            if not n:
                continue
            if n.op_type == "Where" and id(n) == id(where_node):
                # The rewriting will add that to the list of rewritten nodes.
                continue
            if n.op_type == "Softmax":
                if len(g.next_nodes(n.output[0])) != 2:
                    return self.none(node, inspect.currentframe().f_lineno)
                continue
            if g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        mul1: NodeProto,
        transpose_mul1: Optional[NodeProto],
        reshape_mul1: Optional[NodeProto],
        gqa_unsqueeze: Optional[NodeProto],
        mul2: NodeProto,
        reshape_mul2: Optional[NodeProto],
        gqa_expand: Optional[NodeProto],
        gqa_reshape: Optional[NodeProto],
        transpose: Optional[NodeProto],
        mat_qk: NodeProto,
        where_node: NodeProto,
        add_node: Optional[NodeProto],
        softmax: NodeProto,
        isnan: NodeProto,
        where: NodeProto,
        gqa_unsqueeze_v: Optional[NodeProto],
        gqa_expand_v: Optional[NodeProto],
        gqa_reshape_v: Optional[NodeProto],
        mat_qkv: NodeProto,
    ) -> List[NodeProto]:
        itype = g.get_type(mul1.input[1])
        suffix = []

        index_inf = self._find_index_inf(g, where_node)
        assert index_inf, (
            f"Could not any inf in node {g.pretty_text(where_node)}, "
            f"the pattern {self.__class__.__name__} should not have matched."
        )
        switch_where = index_inf == 1
        if switch_where:
            suffix.append("SW")

        if transpose is None:
            assert (
                mat_qk.op_type == "FusedMatMul"
            ), f"transpose is None but mat_qk={g.pretty_node(mat_qk)}"
            suffix.append("NoT")
        if gqa_reshape:
            gqa = "GQA" if gqa_reshape.op_type == "Reshape" else "GQAsQ"
            gqa_args = [gqa_expand.input[1], gqa_reshape.input[1]]
        else:
            gqa = ""
            gqa_args = []

        # nodes to add
        attention_nodes = []
        if g.is_used_more_than_once(where_node.output[0]):
            # keep it if it used more than once
            attention_nodes.append(where_node)

        scale = mul1.input[1]
        if reshape_mul1 is not None:
            assert (
                reshape_mul2 is not None
                and transpose_mul1 is not None
                and transpose is not None
                and gqa_unsqueeze is None
            ), (
                f"Inconsistencies with {reshape_mul2=}, {transpose_mul1=}, "
                f"{transpose=}, {gqa_unsqueeze=}"
            )
            keys = g.unique_name(f"{self.__class__.__name__}--{mul1.input[0]}")
            values = g.unique_name(f"{self.__class__.__name__}--{mul2.input[0]}")
            keys_t = g.unique_name(f"{self.__class__.__name__}--{transpose_mul1.input[0]}")
            values_t = g.unique_name(f"{self.__class__.__name__}--{transpose.input[0]}")
            attention_nodes.extend(
                [
                    g.make_node(
                        "Reshape",
                        [mul1.input[0], reshape_mul1.input[1]],
                        [keys_t],
                        name=f"{self.__class__.__name__}--{reshape_mul1.name}",
                    ),
                    g.make_node(
                        "Reshape",
                        [mul2.input[0], reshape_mul2.input[1]],
                        [values_t],
                        name=f"{self.__class__.__name__}--{reshape_mul2.name}",
                    ),
                    g.make_node(
                        "Transpose",
                        [keys_t],
                        [keys],
                        perm=[0, 2, 1, 3],
                        name=f"{self.__class__.__name__}--{transpose_mul1.name}",
                    ),
                    g.make_node(
                        "Transpose",
                        [values_t],
                        [values],
                        perm=[0, 2, 1, 3],
                        name=f"{self.__class__.__name__}--{transpose.name}",
                    ),
                ]
            )
        else:
            keys = mul1.input[0]
            values = gqa_unsqueeze.input[0] if gqa_reshape else mul2.input[0]

        name = f"{self._operator_name}{gqa}{''.join(suffix)}_to{itype}"
        attention_nodes.append(
            g.make_node(
                name,
                [
                    keys,
                    values,
                    gqa_unsqueeze_v.input[0] if gqa_reshape else mat_qkv.input[1],
                    where_node.input[0],
                    scale,
                    *gqa_args,
                ],
                [mat_qkv.output[0]],
                name=f"{self.__class__.__name__}--{softmax.name}",
                domain=self._domain_name,
            )
        )

        nodes_to_return = attention_nodes

        # Creates the local function
        if not g.builder.has_local_function(name, domain=self._domain_name):
            self._add_local_function(
                g.builder,
                name,
                itype=itype,
                gqa=gqa,
                switch_where=switch_where,
                use_qga_squeeze=gqa_reshape and gqa_reshape.op_type == "Squeeze",
            )
        return nodes_to_return

    @classmethod
    def _add_local_function(
        cls,
        g: GraphBuilder,
        name: str,
        itype: int,
        gqa: bool,
        switch_where: bool,
        use_qga_squeeze: bool,
    ):
        lg = GraphBuilder(g.main_opset, as_function=True)
        lg.make_tensor_input("query")
        lg.make_tensor_input("keys")
        lg.make_tensor_input("values")
        mask_name = "not_mask" if switch_where else "mask"
        lg.make_tensor_input(mask_name)
        lg.make_tensor_input("scale_sqrt")

        scaled_keys = lg.op.Mul("keys", "scale_sqrt", name=cls.__name__)
        if gqa:
            lg.make_tensor_input("expand_shape")
            lg.make_tensor_input("gqa_shape")

            two = np.array([2], dtype=np.int64)
            unsq_keys = lg.op.UnsqueezeAnyOpset(scaled_keys, two, name=cls.__name__)
            unsq_values = lg.op.UnsqueezeAnyOpset("values", two, name=cls.__name__)
            exp_keys = lg.op.Expand(unsq_keys, "expand_shape")
            exp_values = lg.op.Expand(unsq_values, "expand_shape")
            if use_qga_squeeze:
                resh_keys = lg.op.Squeeze(exp_keys, "gqa_shape")
                resh_values = lg.op.Squeeze(exp_values, "gqa_shape")
            else:
                resh_keys = lg.op.Reshape(exp_keys, "gqa_shape")
                resh_values = lg.op.Reshape(exp_values, "gqa_shape")
            scaled_keys = resh_keys
            values = resh_values
        else:
            values = "values"

        scaled_query = lg.op.Mul("query", "scale_sqrt", name=cls.__name__)
        scaled_keys_t = lg.op.Transpose(scaled_keys, perm=(0, 1, 3, 2), name=cls.__name__)
        qk = lg.op.MatMul(scaled_query, scaled_keys_t, name=cls.__name__)
        dtype = tensor_dtype_to_np_dtype(itype)
        zero = np.array([0], dtype=dtype)
        minfty = np.array([-np.inf], dtype=dtype)
        where_args = (minfty, qk) if switch_where else (qk, minfty)
        masked_qk = lg.op.Where(mask_name, *where_args, name=cls.__name__)
        softmax = lg.op.Softmax(masked_qk, axis=-1, name=cls.__name__)
        filtered = lg.op.Where(
            lg.op.IsNaN(softmax, name=cls.__name__), zero, softmax, name=cls.__name__
        )
        lg.op.MatMul(filtered, values, outputs=["Y"], name=cls.__name__)

        lg.make_tensor_output("Y")

        function_options = FunctionOptions(
            export_as_function=True,
            name=name,
            domain=cls._domain_name,
            move_initializer_to_constant=True,
        )
        g.make_local_function(lg, function_options=function_options)
        assert g.has_local_function(
            name, domain=cls._domain_name
        ), f"The function {cls._domain_name}.{name} was not added to the builder."


class _CommonGQAMethods:
    def _match_keys_or_values(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        keys_or_values: str,
    ) -> Optional[Tuple[NodeProto, NodeProto, NodeProto, Tuple[Tuple[Union[int, str], ...]]]]:

        gqa_reshape = g.node_before(keys_or_values)
        if (
            not gqa_reshape
            or gqa_reshape.op_type not in ("Reshape", "Squeeze")
            or gqa_reshape.domain != ""
            or g.main_opset < 18
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        gqa_expand = g.node_before(gqa_reshape.input[0])
        if gqa_expand.op_type != "Expand":
            return self.none(node, inspect.currentframe().f_lineno)

        gqa_unsqueeze = g.node_before(gqa_expand.input[0])
        if gqa_unsqueeze.op_type != "Unsqueeze":
            return self.none(node, inspect.currentframe().f_lineno)
        #
        if not g.is_constant(gqa_expand.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        exp_shape = g.get_computed_constant(gqa_expand.input[1])
        if tuple(exp_shape[:2]) != (1, 1) or tuple(exp_shape[3:]) != (1, 1):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(gqa_unsqueeze.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        unsq_shape = g.get_computed_constant(gqa_unsqueeze.input[1])
        if tuple(unsq_shape) != (2,):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(gqa_reshape.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        resh_shape = g.get_computed_constant(gqa_reshape.input[1])
        if gqa_reshape.op_type == "Reshape":
            if resh_shape.size != 4:
                return self.none(node, inspect.currentframe().f_lineno)
        elif gqa_reshape.op_type == "Squeeze":
            if resh_shape.size != 1:
                return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(gqa_unsqueeze.input[0]) or not g.has_shape(gqa_reshape.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape1 = g.get_shape_renamed(gqa_unsqueeze.input[0])
        shape2 = g.get_shape_renamed(gqa_reshape.output[0])
        if shape1[0] != shape2[0] or shape1[2] != shape2[2] or shape1[3] != shape2[3]:
            return self.none(node, inspect.currentframe().f_lineno)

        return (
            gqa_unsqueeze,
            gqa_expand,
            gqa_reshape,
            (tuple(unsq_shape), tuple(exp_shape), tuple(resh_shape)),
        )


class FunctionAttentionGQAPattern(FunctionAttentionPattern, _CommonGQAMethods):
    """
    Merges onnx nodes equivalent to repeat interleave followed by function
    ``LocalAttention`` into ``LocalAttentionGQA`` (GQA for GroupQueryAttention).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_cat(["cat FLOAT(batch, 4, past_length+seq_length, 32)"])
            I_init1_s___RSh1(["init1_s_::RSh1 FLOAT(1)"])
            I_to(["to BOOL(seq_length, total_length)"])
            I_init7_s4_0_8__1_32(["init7_s4_0_8_-1_32 INT64(4)"])
            I_init7_s5_1_1_2_1_1(["init7_s5_1_1_2_1_1 INT64(5)"])
            I_cat_1(["cat_1 FLOAT(batch, 4, past_length+seq_length, 32)"])
            I_query(["query FLOAT(batch, 8, seq_length, 32)"])

            Constant_0[["Constant() -#gt; init7_s5_1_1_2_1_1"]]
            Constant_1[["Constant() -#gt; init7_s4_0_8_-1_32"]]
            Constant_2[["Constant() -#gt; init1_s_::RSh1"]]
            Unsqueeze_3[["Unsqueeze(., [2])"]]
            Expand_4[["Expand(., .)"]]
            Reshape_5[["Reshape(., .)"]]
            Unsqueeze_6[["Unsqueeze(., [2])"]]
            Expand_7[["Expand(., .)"]]
            Reshape_8[["Reshape(., .)"]]
            LocalAttentionSW_to1_9[["intermediate.LocalAttentionSW_to1(., ., ., ., .)"]]

            I_cat -->|"FLOAT(batch, 4, past_length+seq_length, 32)"| Unsqueeze_3
            Unsqueeze_3 --> Expand_4
            Constant_0 -->|"INT64(5)"| Expand_4
            Expand_4 --> Reshape_5
            Constant_1 -->|"INT64(4)"| Reshape_5
            I_cat_1 -->|"FLOAT(batch, 4, past_length+seq_length, 32)"| Unsqueeze_6
            Unsqueeze_6 --> Expand_7
            Constant_0 -->|"INT64(5)"| Expand_7
            Expand_7 --> Reshape_8
            Constant_1 -->|"INT64(4)"| Reshape_8
            I_query -->|"FLOAT(batch, 8, seq_length, 32)"| LocalAttentionSW_to1_9
            Reshape_5 --> LocalAttentionSW_to1_9
            Reshape_8 --> LocalAttentionSW_to1_9
            I_to -->|"BOOL(seq_length, total_length)"| LocalAttentionSW_to1_9
            Constant_2 -->|"FLOAT(1)"| LocalAttentionSW_to1_9

            O_output_0(["output_0 FLOAT(batch, 8, seq_length, 32)"])
            LocalAttentionSW_to1_9 --> O_output_0

            class I_cat,I_init1_s___RSh1,I_to,I_init7_s4_0_8__1_32 ioNode
            class I_init7_s5_1_1_2_1_1,I_cat_1,I_query,O_output_0 ioNode
            class Constant_0,Constant_1,Constant_2 constNode
            class Unsqueeze_3,Expand_4,Reshape_5,Unsqueeze_6,Expand_7 opNode
            class Reshape_8,LocalAttentionSW_to1_9 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_cat(["cat FLOAT(batch, 4, past_length+seq_length, 32)"])
            I_init1_s___RSh1(["init1_s_::RSh1 FLOAT(1)"])
            I_to(["to BOOL(seq_length, total_length)"])
            I_init7_s4_0_8__1_32(["init7_s4_0_8_-1_32 INT64(4)"])
            I_init7_s5_1_1_2_1_1(["init7_s5_1_1_2_1_1 INT64(5)"])
            I_cat_1(["cat_1 FLOAT(batch, 4, past_length+seq_length, 32)"])
            I_query(["query FLOAT(batch, 8, seq_length, 32)"])

            LocalAttentionGQASW_to1_0[["intermediate.LocalAttentionGQASW_to1(
            ., ., ., ., ., ., .)"]]

            I_query -->|"FLOAT(batch, 8, seq_length, 32)"| LocalAttentionGQASW_to1_0
            I_cat -->|"FLOAT(batch, 4, past_length+seq_length, 32)"| LocalAttentionGQASW_to1_0
            I_cat_1 -->|"FLOAT(batch, 4, past_length+seq_length, 32)"| LocalAttentionGQASW_to1_0
            I_to -->|"BOOL(seq_length, total_length)"| LocalAttentionGQASW_to1_0
            I_init1_s___RSh1 -->|"FLOAT(1)"| LocalAttentionGQASW_to1_0
            I_init7_s5_1_1_2_1_1 -->|"INT64(5)"| LocalAttentionGQASW_to1_0
            I_init7_s4_0_8__1_32 -->|"INT64(4)"| LocalAttentionGQASW_to1_0

            O_output_0(["output_0 FLOAT(batch, 8, seq_length, 32)"])
            LocalAttentionGQASW_to1_0 --> O_output_0

            class I_cat,I_init1_s___RSh1,I_to,I_init7_s4_0_8__1_32 ioNode
            class I_init7_s5_1_1_2_1_1,I_cat_1,I_query,O_output_0 ioNode
            class LocalAttentionGQASW_to1_0 opNode
    """

    _operator_gqa_name = f"{FunctionAttentionPattern._operator_name}GQA"

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            not node.op_type.startswith(FunctionAttentionPattern._operator_name)
            or node.op_type.startswith(FunctionAttentionGQAPattern._operator_gqa_name)
            or node.domain != FunctionAttentionGQAPattern._domain_name
        ):
            return self.none()

        keys, values = node.input[1:3]

        matched_keys = self._match_keys_or_values(g, node, keys)
        if not matched_keys:
            return self.none(node, inspect.currentframe().f_lineno)

        matched_values = self._match_keys_or_values(g, node, values)
        if not matched_values:
            return self.none(node, inspect.currentframe().f_lineno)

        gqa_unsqueeze, gqa_expand, gqa_reshape, shapes = matched_keys
        gqa_unsqueeze_v, gqa_expand_v, gqa_reshape_v, _shapes_v = matched_values

        unsq_shape, exp_shape, resh_shape = shapes
        unsq_shape_v, exp_shape_v, resh_shape_v = shapes

        if unsq_shape_v != unsq_shape:
            return self.none(node, inspect.currentframe().f_lineno)
        if exp_shape != exp_shape_v:
            return self.none(node, inspect.currentframe().f_lineno)
        if resh_shape_v != resh_shape:
            return self.none(node, inspect.currentframe().f_lineno)

        # Final verification, let's check none the nodes is used outside the pattern.
        nodes = [
            gqa_unsqueeze,
            gqa_expand,
            gqa_reshape,
            gqa_unsqueeze_v,
            gqa_expand_v,
            gqa_reshape_v,
            node,
        ]
        for n in nodes[:-1]:
            if n and g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        gqa_unsqueeze: NodeProto,
        gqa_expand: NodeProto,
        gqa_reshape: NodeProto,
        gqa_unsqueeze_v: NodeProto,
        gqa_expand_v: NodeProto,
        gqa_reshape_v: NodeProto,
        attn: NodeProto,
    ) -> List[NodeProto]:
        itype = g.get_type(gqa_unsqueeze.input[0])
        gqa = "" if gqa_reshape.op_type == "Reshape" else "sQ"
        name = f"{self._operator_gqa_name}{gqa}{attn.op_type[len(self._operator_name):]}"
        attention_nodes = [
            g.make_node(
                name,
                [
                    attn.input[0],
                    gqa_unsqueeze.input[0],
                    gqa_unsqueeze_v.input[0],
                    attn.input[3] if len(attn.input) > 3 else "",
                    attn.input[4] if len(attn.input) > 4 else "",
                    gqa_expand.input[1],
                    gqa_reshape.input[1],
                ],
                [attn.output[0]],
                name=f"{self.__class__.__name__}--{attn.name}",
                domain=self._domain_name,
            )
        ]

        # Creates the local function
        if not g.builder.has_local_function(name, domain=self._domain_name):
            self._add_local_function(
                g.builder,
                name,
                itype=itype,
                gqa=True,
                switch_where="SW" in attn.op_type,
                use_qga_squeeze=gqa_reshape_v.op_type == "Squeeze",
            )
        return attention_nodes


class AttentionGQAPattern(PatternOptimization, _CommonGQAMethods):
    """
    Fuses LocalAttention into Attention.
    Opset must be >= 23 to do so.

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_key(["key FLOAT(a, 2, c, 8)"])
            I_mask(["mask BOOL(a, 1, c, c+h)"])
            I_value(["value FLOAT(a, 2, c, 8)"])
            I_past_key(["past_key FLOAT(a, 2, h, 8)"])
            I_query(["query FLOAT(a, 4, c, 8)"])
            I_past_value(["past_value FLOAT(a, 2, h, 8)"])

            Concat_0[["Concat(., ., axis=2)"]]
            Concat_1[["Concat(., ., axis=2)"]]
            Unsqueeze_2[["Unsqueeze(., [2])"]]
            Expand_3[["Expand(., [1, 1, 2, 1, 1])"]]
            Reshape_4[["Reshape(., [0, 4, -1, 8])"]]
            Unsqueeze_5[["Unsqueeze(., [2])"]]
            Expand_6[["Expand(., [1, 1, 2, 1, 1])"]]
            Reshape_7[["Reshape(., [0, 4, -1, 8])"]]
            Attention_8[["Attention(., ., ., .)"]]

            I_past_key -->|"FLOAT(a, 2, h, 8)"| Concat_0
            I_key -->|"FLOAT(a, 2, c, 8)"| Concat_0
            I_past_value -->|"FLOAT(a, 2, h, 8)"| Concat_1
            I_value -->|"FLOAT(a, 2, c, 8)"| Concat_1
            Concat_0 -->|"FLOAT(a, 2, c+h, 8)"| Unsqueeze_2
            Unsqueeze_2 -->|"FLOAT(a, 2, 1, c+h, 8)"| Expand_3
            Expand_3 -->|"FLOAT(a, 2, 2, c+h, 8)"| Reshape_4
            Concat_1 -->|"FLOAT(a, 2, c+h, 8)"| Unsqueeze_5
            Unsqueeze_5 -->|"FLOAT(a, 2, 1, c+h, 8)"| Expand_6
            Expand_6 -->|"FLOAT(a, 2, 2, c+h, 8)"| Reshape_7
            I_query -->|"FLOAT(a, 4, c, 8)"| Attention_8
            Reshape_4 -->|"FLOAT(a, 4, c+h, 8)"| Attention_8
            Reshape_7 -->|"FLOAT(a, 4, c+h, 8)"| Attention_8
            I_mask -->|"BOOL(a, 1, c, c+h)"| Attention_8

            O_present_value(["present_value FLOAT(a, 2, c+h, 8)"])
            Concat_1 --> O_present_value
            O_present_key(["present_key FLOAT(a, 2, c+h, 8)"])
            Concat_0 --> O_present_key
            O_Y(["Y FLOAT(a, 4, c_, 8)"])
            Attention_8 --> O_Y

            class I_key,I_mask,I_value,I_past_key,I_query,I_past_value ioNode
            class O_present_value,O_present_key,O_Y ioNode
            class Concat_0,Concat_1,Unsqueeze_2,Expand_3,Reshape_4,Unsqueeze_5 opNode
            class Expand_6,Reshape_7,Attention_8 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_key(["key FLOAT(a, 2, c, 8)"])
            I_mask(["mask BOOL(a, 1, c, c+h)"])
            I_value(["value FLOAT(a, 2, c, 8)"])
            I_past_key(["past_key FLOAT(a, 2, h, 8)"])
            I_query(["query FLOAT(a, 4, c, 8)"])
            I_past_value(["past_value FLOAT(a, 2, h, 8)"])

            Attention_0[["Attention(., ., ., ., ., .)"]]

            I_query -->|"FLOAT(a, 4, c, 8)"| Attention_0
            I_key -->|"FLOAT(a, 2, c, 8)"| Attention_0
            I_value -->|"FLOAT(a, 2, c, 8)"| Attention_0
            I_mask -->|"BOOL(a, 1, c, c+h)"| Attention_0
            I_past_key -->|"FLOAT(a, 2, h, 8)"| Attention_0
            I_past_value -->|"FLOAT(a, 2, h, 8)"| Attention_0

            O_present_value(["present_value FLOAT(a, 2, c+h, 8)"])
            Attention_0 --> O_present_value
            O_present_key(["present_key FLOAT(a, 2, c+h, 8)"])
            Attention_0 --> O_present_key
            O_Y(["Y FLOAT(a, 4, c_, 8)"])
            Attention_0 --> O_Y

            class I_key,I_mask,I_value,I_past_key,I_query,I_past_value opNode
            class O_present_value,O_present_key,O_Y ioNode
            class Attention_0 opNode
    """

    _prefixes_operator_name = (
        f"{FunctionAttentionGQAPattern._operator_gqa_name}SW_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}SWsQ_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}sQ_to",
    )

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if g.main_opset < 23:
            return self.none()
        if (
            (node.op_type != "Attention" or node.domain != "")
            and (
                not node.op_type.startswith(self._prefixes_operator_name)
                or node.domain != FunctionAttentionGQAPattern._domain_name
                or len(node.input) != 7
            )
        ) or len(node.output) > 1:
            return self.none()

        if len(node.input) > 3 and (
            not g.has_rank(node.input[3]) or g.get_rank(node.input[3]) < 2
        ):
            # Only 2D ranks allowed.
            return self.none(node, inspect.currentframe().f_lineno)

        if node.op_type == "Attention":
            if not g.has_rank(node.input[0]) and g.get_rank(node.input[0]) != 4:
                # Only 4D Attention
                return self.none(node, inspect.currentframe().f_lineno)
            # Node Attention, we still need to check if there is some GQA node.
            gqa_keys = self._match_keys_or_values(g, node, node.input[1])
            if not gqa_keys:
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_values = self._match_keys_or_values(g, node, node.input[2])
            if not gqa_values:
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_unsqueeze, gqa_expand, gqa_reshape, shapes = gqa_keys
            gqa_unsqueeze_v, gqa_expand_v, gqa_reshape_v, shapes_v = gqa_values
            unsq_shape, exp_shape, resh_shape = shapes
            unsq_shape_v, exp_shape_v, resh_shape_v = shapes_v

            if unsq_shape_v != unsq_shape:
                return self.none(node, inspect.currentframe().f_lineno)
            if exp_shape != exp_shape_v:
                return self.none(node, inspect.currentframe().f_lineno)
            if resh_shape_v != resh_shape:
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_nodes = [
                gqa_unsqueeze,
                gqa_expand,
                gqa_reshape,
                gqa_unsqueeze_v,
                gqa_expand_v,
                gqa_reshape_v,
            ]

            concats = g.node_before(gqa_unsqueeze.input[0]), g.node_before(
                gqa_unsqueeze_v.input[0]
            )
            if None in concats:
                return self.none(node, inspect.currentframe().f_lineno)
            if len(concats[0].input) != 2 or len(concats[1].input) != 2:
                return self.none(node, inspect.currentframe().f_lineno)
            if concats[0].op_type != "Concat" or concats[1].op_type != "Concat":
                return self.none(node, inspect.currentframe().f_lineno)
            if g.get_attribute_with_default(
                concats[0], "axis", 0
            ) != g.get_attribute_with_default(concats[1], "axis", 0):
                return self.none(node, inspect.currentframe().f_lineno)

        else:
            keys, values = node.input[1:3]
            concats = g.node_before(keys), g.node_before(values)
            if None in concats:
                return self.none(node, inspect.currentframe().f_lineno)
            if len(concats[0].input) != 2 or len(concats[1].input) != 2:
                return self.none(node, inspect.currentframe().f_lineno)
            if concats[0].op_type != "Concat" or concats[1].op_type != "Concat":
                return self.none(node, inspect.currentframe().f_lineno)
            if g.get_attribute_with_default(
                concats[0], "axis", 0
            ) != g.get_attribute_with_default(concats[1], "axis", 0):
                return self.none(node, inspect.currentframe().f_lineno)

            # Local function
            if not g.is_constant_scalar(node.input[4]):
                return self.none(node, inspect.currentframe().f_lineno)

            if not g.is_constant(node.input[5]):
                return self.none(node, inspect.currentframe().f_lineno)
            cst = g.get_computed_constant(node.input[5])
            if cst is None:
                return self.none(node, inspect.currentframe().f_lineno)
            cst = tuple(cst)
            if len(cst) < 4:
                return self.none(node, inspect.currentframe().f_lineno)
            if cst[:2] != cst[3:] or cst[:2] != (1, 1):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant(node.input[6]):
                return self.none(node, inspect.currentframe().f_lineno)
            shape_or_axis = g.get_computed_constant(node.input[6])
            if shape_or_axis is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if "sQ_to" in node.op_type:
                # This is an axis for a Squeeze node.
                if not g.get_shape(node.input[1]):
                    # We need that shape to get kv_num_heads.
                    return self.none(node, inspect.currentframe().f_lineno)
            else:
                # This is a shape for a Reshape node.
                if shape_or_axis[1] <= 0:
                    return self.none(node, inspect.currentframe().f_lineno)
            gqa_nodes = [None for _ in range(6)]

        # Final verification, let's check none the nodes is used outside the pattern.
        nodes = [*concats, *gqa_nodes, node]
        for n in nodes[2:-1]:
            if n and g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, nodes, self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        keys_concat_node: NodeProto,
        values_concat_node: NodeProto,
        gqa_unsqueeze: Optional[NodeProto],
        gqa_expand: Optional[NodeProto],
        gqa_reshape: Optional[NodeProto],
        gqa_unsqueeze_v: Optional[NodeProto],
        gqa_expand_v: Optional[NodeProto],
        gqa_reshape_v: Optional[NodeProto],
        local_attention_gqa: Optional[NodeProto],
    ) -> List[NodeProto]:
        query, _keys, _values, mask = local_attention_gqa.input[:4]
        attn_kwargs = {}
        if local_attention_gqa.op_type == "Attention":
            scale = g.get_attribute_with_default(local_attention_gqa, "scale", None)
            if scale is not None:
                attn_kwargs["scale"] = scale
            attn_kwargs["is_causal"] = g.get_attribute_with_default(
                local_attention_gqa, "is_causal", 0
            )
        else:
            scale = g.get_constant_scalar(local_attention_gqa.input[4]) ** 2  # this scale ** 0.5
            attn_kwargs["scale"] = scale

        # In case we need the 3D pattern.
        # expand_shape = g.get_computed_constant(local_attention_gqa.input[5])
        # repeat = int(expand_shape[2])
        # if "sQ_" in local_attention_gqa.op_type:
        #    k_shape = g.get_shape(local_attention_gqa.input[1])
        #    kv_num_heads = k_shape[1]
        # else:
        #    reshape_shape = g.get_computed_constant(local_attention_gqa.input[6])
        #    kv_num_heads = reshape_shape[1] // repeat
        #
        # num_heads = kv_num_heads * repeat

        nodes = []

        final_mask = mask
        if mask:
            switch_where = "SW" in local_attention_gqa.op_type
            if switch_where:
                # mask is not mask if SW
                if g.get_type(mask) == TensorProto.BOOL:
                    final_mask = g.unique_name(f"{self.__class__.__name__}--{mask}")
                    nodes.append(g._make_node("Not", [mask], [final_mask]))
                else:
                    raise NotImplementedError(
                        f"float mask is not implemented yet for pattern "
                        f"{self.__class__.__name__!r}"
                    )

        nodes.extend(
            [
                g._make_node(
                    "Attention",
                    [
                        query,
                        keys_concat_node.input[1],
                        values_concat_node.input[1],
                        final_mask,
                        keys_concat_node.input[0],
                        values_concat_node.input[0],
                    ],
                    [
                        local_attention_gqa.output[0],
                        keys_concat_node.output[0],
                        values_concat_node.output[0],
                    ],
                    # q_num_heads=num_heads,
                    # kv_num_heads=kv_num_heads,
                    **attn_kwargs,
                )
            ]
        )
        for node in nodes:
            if not node.name:
                node.name = g.builder.unique_node_name(
                    f"{self.__class__.__name__}--{local_attention_gqa.name}"
                )
        return nodes
