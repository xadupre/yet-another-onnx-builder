import inspect
from typing import List, Optional
import numpy as np
import onnx.numpy_helper as onh
from onnx import NodeProto, TensorProto
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ..patterns_api import MatchResult, PatternOptimization


class SimplifiedLayerNormalizationPattern(PatternOptimization):
    """
    Fuses the nodes equivalent to SimplifiedLayerNormalization.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, D)"])
            I_axis(["axis INT64(1)"])

            Constant_0[["Constant() -#gt; axis"]]
            Pow_1[["Pow(., [2.0])"]]
            ReduceMean_2[["ReduceMean(., .)"]]
            Add_3[["Add(., [1e-06])"]]
            Sqrt_4[["Sqrt(.)"]]
            Div_5[["Div([1.0], .)"]]
            Mul_6[["Mul(., .)"]]

            I_X -->|"FLOAT(a, D)"| Pow_1
            Pow_1 -->|"FLOAT(a, D)"| ReduceMean_2
            Constant_0 -->|"INT64(1)"| ReduceMean_2
            ReduceMean_2 -->|"FLOAT(a, 1)"| Add_3
            Add_3 -->|"FLOAT(a, 1)"| Sqrt_4
            Sqrt_4 -->|"FLOAT(a, 1)"| Div_5
            Div_5 -->|"FLOAT(a, 1)"| Mul_6
            I_X -->|"FLOAT(a, D)"| Mul_6

            O_Z(["Z FLOAT(a, 1)"])
            Div_5 --> O_Z
            O_Y(["Y FLOAT(a, D)"])
            Mul_6 --> O_Y

            class I_X,I_axis,O_Z,O_Y ioNode
            class Constant_0 constNode
            class Pow_1,ReduceMean_2,Add_3,Sqrt_4,Div_5,Mul_6 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, D)"])
            I_axis(["axis INT64(1)"])

            Shape_0[["Shape(.)"]]
            Gather_1[["Gather(., .)"]]
            ConstantOfShape_2[["ConstantOfShape(.)"]]
            SimplifiedLayerNormalization_3[["SimplifiedLayerNormalization(., ., axis=-1, stash_type=1)"]]

            I_X -->|"FLOAT(a, D)"| Shape_0
            Shape_0 -->|"INT64(2)"| Gather_1
            I_axis -->|"INT64(1)"| Gather_1
            Gather_1 -->|"INT64(1)"| ConstantOfShape_2
            I_X -->|"FLOAT(a, D)"| SimplifiedLayerNormalization_3
            ConstantOfShape_2 --> SimplifiedLayerNormalization_3

            O_Z(["Z FLOAT(a, 1)"])
            SimplifiedLayerNormalization_3 --> O_Z
            O_Y(["Y FLOAT(a, D)"])
            SimplifiedLayerNormalization_3 --> O_Y

            class I_X,I_axis,O_Z,O_Y ioNode
            class Shape_0,Gather_1,ConstantOfShape_2,SimplifiedLayerNormalization_3 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ReduceMean" or node.domain != "":
            return self.none()
        if len(node.input) < 2:
            return self.none(node, inspect.currentframe().f_lineno)

        axis = g.get_constant_or_attribute(node, "axes", input_index=1, cvt=tuple)
        assert isinstance(axis, tuple), f"unexpected type {type(axis)} for axis"
        if len(axis) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        node_pow = g.node_before(node.input[0])
        if node_pow is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if node_pow.op_type != "Pow" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node_pow.input[1], 2):
            return self.none(node, inspect.currentframe().f_lineno)

        nodes_add = g.next_nodes(node.output[0])
        if len(nodes_add) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        node_add = nodes_add[0]
        if node_add.op_type != "Add" or node_add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(node_add.input[0]) and not g.is_constant_scalar(
            node_add.input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        node_sqrt = g.next_node(node_add.output[0])
        if node_sqrt.op_type != "Sqrt" or node_sqrt.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        node_reciprocal = g.next_node(node_sqrt.output[0])
        if node_reciprocal.op_type not in ("Reciprocal", "Div") or node_reciprocal.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if node_reciprocal.op_type == "Div":
            if node_reciprocal.input[1] != node_sqrt.output[0]:
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant_scalar(node_reciprocal.input[0], 1):
                return self.none(node, inspect.currentframe().f_lineno)

        node_mul = g.next_node(node_reciprocal.output[0])
        if node_mul.op_type != "Mul" or node_mul.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if (
            g.is_used_more_than_once(node_pow.output[0])
            or g.is_used_more_than_once(node.output[0])
            or g.is_used_more_than_once(node_add.output[0])
            or g.is_used_more_than_once(node_sqrt.output[0])
        ):
            # intermediate results are used
            return self.none(node, inspect.currentframe().f_lineno)

        mul_i = set(node_mul.input)
        cmp = {node_pow.input[0], node_reciprocal.output[0]}
        if mul_i != cmp:
            # We check the multiplication node takes the output of the div node
            # and the input of the pow node.
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [node_pow, node, node_add, node_sqrt, node_reciprocal, node_mul]
        return MatchResult(self, nodes, self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_pow: NodeProto,
        node_reduce: NodeProto,
        node_add: NodeProto,
        node_sqrt: NodeProto,
        node_reciprocal: NodeProto,
        node_mul: NodeProto,
    ) -> List[NodeProto]:
        nname = node_reduce.name
        nodes = []
        epsilon = g.get_computed_constant(node_add.input[1])
        shape = g.get_shape(node_reduce.input[0]) if g.has_shape(node_reduce.input[0]) else None
        axis = g.get_constant_or_attribute(node_reduce, "axes", input_index=1)[0]
        assert shape is None or axis < len(
            shape
        ), f"axis={axis} and shape={shape} don't match for {node_reduce.input[0]!r}"
        stash_type = g.get_type(node_reduce.input[0])
        dtype = tensor_dtype_to_np_dtype(stash_type)
        if shape is not None and isinstance(shape[axis], int):
            # a constant
            scale = g.make_initializer(
                f"ONES{shape[axis]}",
                np.ones((shape[axis],), dtype=dtype),
                source="SimplifiedLayerNormalizationPattern.apply.scale.1",
            )
        else:
            sh = g.make_node(
                "Shape", [node_pow.input[0]], name=f"{self.__class__.__name__}--{nname}"
            )
            axis_name = g.make_initializer(
                "",
                np.array([axis], dtype=np.int64),
                source="SimplifiedLayerNormalizationPattern.apply.axis",
            )
            ga = g.make_node(
                "Gather", [sh.output[0], axis_name], name=f"{self.__class__.__name__}--{nname}"
            )
            # sc = g.make_node_check_opset(
            #    "Unsqueeze", [ga.output[0]], axes=[0],
            #       name=f"{self.__class__.__name__}--{nname}"
            # )
            cc = g.make_node(
                "ConstantOfShape",
                [ga.output[0]],
                value=onh.from_array(np.array([1], dtype=dtype)),
                name=f"{self.__class__.__name__}--{nname}",
            )
            scale = cc.output[0]
            nodes.extend([sh, ga, cc])

        layer = g.make_node(
            "SimplifiedLayerNormalization",
            [node_pow.input[0], scale],
            [node_mul.output[0], node_reciprocal.output[0]],
            epsilon=float(epsilon[0] if epsilon.shape else epsilon),
            axis=int(axis),
            stash_type=stash_type,
            name=f"{self.__class__.__name__}--{nname}",
        )

        nodes.append(layer)
        return nodes


class SkipLayerNormalizationPattern(PatternOptimization):
    """
    Replaces the sequence Add + LayerNormalization into SkipLayerNormalization.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X2(["X2 FLOAT16(a, b, c)"])
            I_X1(["X1 FLOAT16(a, b, c)"])
            I_scale(["scale FLOAT16(c)"])
            I_bias(["bias FLOAT16(c)"])

            Add_0[["Add(., .)"]]
            LayerNormalization_1[["LayerNormalization(., ., ., axis=-1)"]]

            I_X1 -->|"FLOAT16(a, b, c)"| Add_0
            I_X2 -->|"FLOAT16(a, b, c)"| Add_0
            Add_0 -->|"FLOAT16(a, b, c)"| LayerNormalization_1
            I_scale -->|"FLOAT16(c)"| LayerNormalization_1
            I_bias -->|"FLOAT16(c)"| LayerNormalization_1

            O_add(["add FLOAT16(a, b, c)"])
            Add_0 --> O_add
            O_Y(["Y FLOAT16(a, b, c)"])
            LayerNormalization_1 --> O_Y

            class I_X2,I_X1,I_scale,I_bias,O_add,O_Y ioNode
            class Add_0,LayerNormalization_1 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X2(["X2 FLOAT16(a, b, c)"])
            I_X1(["X1 FLOAT16(a, b, c)"])
            I_scale(["scale FLOAT16(c)"])
            I_bias(["bias FLOAT16(c)"])

            SkipLayerNormalization_0[["com.microsoft.SkipLayerNormalization(., ., ., .)"]]

            I_X1 -->|"FLOAT16(a, b, c)"| SkipLayerNormalization_0
            I_X2 -->|"FLOAT16(a, b, c)"| SkipLayerNormalization_0
            I_scale -->|"FLOAT16(c)"| SkipLayerNormalization_0
            I_bias -->|"FLOAT16(c)"| SkipLayerNormalization_0

            O_add(["add FLOAT16(a, b, c)"])
            SkipLayerNormalization_0 --> O_add
            O_Y(["Y FLOAT16(a, b, c)"])
            SkipLayerNormalization_0 --> O_Y

            class I_X2,I_X1,I_scale,I_bias,O_add,O_Y ioNode
            class SkipLayerNormalization_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "LayerNormalization" or node.domain != "":
            return self.none()
        if not g.has_rank(node.input[0]) or g.get_rank(node.input[0]) not in (2, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.input) > 1 and (
            not g.has_rank(node.input[1]) or g.get_rank(node.input[1]) != 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.input) > 2 and (
            not g.has_rank(node.input[2]) or g.get_rank(node.input[2]) != 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute(node, "axis", exc=False)
        axis = 0 if axis is None else axis.i
        if axis != -1:
            return self.none(node, inspect.currentframe().f_lineno)
        before = g.node_before(node.input[0])
        if before is None or before.op_type != "Add":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_rank(before.input[1]) or g.get_rank(before.input[1]) not in (2, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        nodes = [before, node]
        return MatchResult(self, nodes, self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", add_node: NodeProto, norm_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        atts = []
        epsilon = g.get_attribute(norm_node, "epsilon", exc=False)
        if epsilon:
            atts.append(epsilon)
        u1 = (
            g.unique_name("unused")
            if len(norm_node.output) < 2 or not norm_node.output[1]
            else norm_node.output[1]
        )
        u2 = (
            g.unique_name("unused")
            if len(norm_node.output) < 3 or not norm_node.output[2]
            else norm_node.output[2]
        )
        layer = g.make_node(
            "SkipLayerNormalization",
            [*add_node.input, *norm_node.input[1:]],
            [norm_node.output[0], u1, u2, add_node.output[0]],
            name=f"{self.__class__.__name__}--{norm_node.name}",
            domain="com.microsoft",
        )
        if atts:
            layer.attribute.extend(atts)
        return [layer]


class SkipSimplifiedLayerNormalizationPattern(PatternOptimization):
    """
    Replaces the sequence Add + SimplifiedLayerNormalization
    by SkipSimplifiedLayerNormalization.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_scale(["scale FLOAT(192)"])
            I_skip(["skip FLOAT(batch, cache, 192)"])
            I_X(["X FLOAT(batch, cache, 192)"])

            Constant_0[["Constant() -#gt; scale"]]
            Add_1[["Add(., .)"]]
            SimplifiedLayerNormalization_2[["SimplifiedLayerNormalization(., ., axis=-1)"]]

            I_X -->|"FLOAT(batch, cache, 192)"| Add_1
            I_skip -->|"FLOAT(batch, cache, 192)"| Add_1
            Add_1 -->|"FLOAT(batch, cache, 192)"| SimplifiedLayerNormalization_2
            Constant_0 -->|"FLOAT(192)"| SimplifiedLayerNormalization_2

            O_xs(["xs FLOAT(batch, cache, 192)"])
            Add_1 --> O_xs
            O_ym(["ym FLOAT(batch, cache, 192)"])
            SimplifiedLayerNormalization_2 --> O_ym

            class I_scale,I_skip,I_X,O_xs,O_ym ioNode
            class Constant_0 constNode
            class Add_1,SimplifiedLayerNormalization_2 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_scale(["scale FLOAT(192)"])
            I_skip(["skip FLOAT(batch, cache, 192)"])
            I_X(["X FLOAT(batch, cache, 192)"])

            SkipSimplifiedLayerNormalization_0[["com.microsoft.SkipSimplifiedLayerNormalization(., ., .)"]]

            I_X -->|"FLOAT(batch, cache, 192)"| SkipSimplifiedLayerNormalization_0
            I_skip -->|"FLOAT(batch, cache, 192)"| SkipSimplifiedLayerNormalization_0
            I_scale -->|"FLOAT(192)"| SkipSimplifiedLayerNormalization_0

            O_xs(["xs FLOAT(batch, cache, 192)"])
            SkipSimplifiedLayerNormalization_0 --> O_xs
            O_ym(["ym FLOAT(batch, cache, 192)"])
            SkipSimplifiedLayerNormalization_0 --> O_ym

            class I_scale,I_skip,I_X,O_xs,O_ym ioNode
            class SkipSimplifiedLayerNormalization_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "SimplifiedLayerNormalization" or node.domain != "":
            return self.none()
        if len(node.output) > 1 and (len(node.output) != 2 or g.is_used(node.output[1])):
            # second output is used
            return self.none(node, inspect.currentframe().f_lineno)

        # axis
        axis = g.get_attribute(node, "axis", exc=False)
        axis = -1 if axis is None else axis.i
        if axis != -1 and (
            not g.has_rank(node.input[0]) or axis != g.get_rank(node.input[0]) - 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # stash_type
        stash_type = g.get_attribute(node, "stash_type", exc=False)
        stash_type = TensorProto.FLOAT if stash_type is None else stash_type.i
        if stash_type != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        add = g.node_before(node.input[0])
        if add.op_type != "Add" or add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            not g.has_shape(add.input[0])
            or not g.has_shape(add.input[1])
            or g.get_shape(add.input[0]) != g.get_shape(add.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [add, node], self.apply)

    def apply(
        self, g: "GraphBuilder", node_add: NodeProto, node_simplified: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        layer = g.make_node(
            "SkipSimplifiedLayerNormalization",
            [*node_add.input, *node_simplified.input[1:]],
            [node_simplified.output[0], "", "", *node_add.output],
            name=f"{self.__class__.__name__}--{node_simplified.name}",
            domain="com.microsoft",
        )
        layer.attribute.extend(
            att for att in node_simplified.attribute if att.name not in {"axis", "stash_type"}
        )
        return [layer]


class SkipSimplifiedLayerNormalizationMulPattern(PatternOptimization):
    """
    Replaces the sequence SkipSimplifiedLayerNormalization + Mul
    by SkipSimplifiedLayerNormalization.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(batch, cache, 192)"])
            I_skip(["skip FLOAT(batch, cache, 192)"])
            I_weights(["weights FLOAT(192)"])

            Constant_0[["Constant() -#gt; scale"]]
            Constant_1[["Constant() -#gt; weights"]]
            SkipSimplifiedLayerNormalization_2[["com.microsoft.SkipSimplifiedLayerNormalization(., ., .)"]]
            Mul_3[["Mul(., .)"]]

            I_X -->|"FLOAT(batch, cache, 192)"| SkipSimplifiedLayerNormalization_2
            I_skip -->|"FLOAT(batch, cache, 192)"| SkipSimplifiedLayerNormalization_2
            Constant_0 -->|"FLOAT(192)"| SkipSimplifiedLayerNormalization_2
            SkipSimplifiedLayerNormalization_2 -->|"FLOAT(batch, cache, 192)"| Mul_3
            Constant_1 -->|"FLOAT(192)"| Mul_3

            O_a(["a FLOAT(batch, cache, 192)"])
            Mul_3 --> O_a
            O_xs(["xs FLOAT(batch, cache, 192)"])
            SkipSimplifiedLayerNormalization_2 --> O_xs

            class I_X,I_skip,I_weights,O_a,O_xs ioNode
            class Constant_0,Constant_1 constNode
            class SkipSimplifiedLayerNormalization_2,Mul_3 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(batch, cache, 192)"])
            I_skip(["skip FLOAT(batch, cache, 192)"])
            I_weights(["weights FLOAT(192)"])

            SkipSimplifiedLayerNormalization_0[["com.microsoft.SkipSimplifiedLayerNormalization(., ., .)"]]

            I_X -->|"FLOAT(batch, cache, 192)"| SkipSimplifiedLayerNormalization_0
            I_skip -->|"FLOAT(batch, cache, 192)"| SkipSimplifiedLayerNormalization_0
            I_weights -->|"FLOAT(192)"| SkipSimplifiedLayerNormalization_0

            O_a(["a FLOAT(batch, cache, 192)"])
            SkipSimplifiedLayerNormalization_0 --> O_a
            O_xs(["xs FLOAT(batch, cache, 192)"])
            SkipSimplifiedLayerNormalization_0 --> O_xs

            class I_X,I_skip,I_weights,O_a,O_xs ioNode
            class SkipSimplifiedLayerNormalization_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type, node.domain) != ("SkipSimplifiedLayerNormalization", "com.microsoft"):
            return self.none()
        if (len(node.output) > 1 and node.output[1]) or (len(node.output) > 2 and node.output[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.input) != 3:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        mul_nodes = g.next_nodes(node.output[0])
        if len(mul_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        mul_node = mul_nodes[0]
        if mul_node.op_type != "Mul" or mul_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        index_cst = 1 if mul_node.input[0] == node.output[0] else 0
        if not g.has_shape(mul_node.input[index_cst]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_shape_renamed(node.input[2]) != g.get_shape_renamed(mul_node.input[index_cst]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(mul_node.input[index_cst]):
            # not supported yet
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, mul_node], self.apply)

    def apply(
        self, g: "GraphBuilder", skip_simp_node: NodeProto, mul_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        index_cst = 1 if mul_node.input[0] == skip_simp_node.output[0] else 0
        cst_skip = g.get_computed_constant(skip_simp_node.input[2])
        if cst_skip.min() == cst_skip.max() == 1:
            cst_name = mul_node.input[index_cst]
        else:
            cst2 = g.get_computed_constant(mul_node.input[index_cst])
            if cst2.min() == cst2.max() == 1:
                cst_name = skip_simp_node.input[2]
            else:
                cst2 = g.get_computed_constant(mul_node.input[index_cst])
                new_cst = cst_skip * cst2
                cst_name = g.make_initializer(
                    f"{skip_simp_node.input[2]}__{mul_node.input[index_cst]}",
                    new_cst,
                    source=f"{self.__class__.__name__}.gamma",
                )

        new_node = g.make_node(
            skip_simp_node.op_type,
            [*skip_simp_node.input[:2], cst_name],
            [mul_node.output[0], *skip_simp_node.output[1:]],
            name=f"{self.__class__.__name__}--{skip_simp_node.name}",
            domain=skip_simp_node.domain,
        )
        if skip_simp_node.attribute:
            new_node.attribute.extend(skip_simp_node.attribute)
        return [new_node]


class SimplifiedLayerNormalizationMulPattern(PatternOptimization):
    """
    Replaces the sequence SimplifiedLayerNormalization + Mul
    by SimplifiedLayerNormalization.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_xs(["xs FLOAT(batch, cache, 192)"])
            I_weights(["weights FLOAT(192)"])

            Constant_0[["Constant() -#gt; scale"]]
            Constant_1[["Constant() -#gt; weights"]]
            SimplifiedLayerNormalization_2[["SimplifiedLayerNormalization(., ., axis=-1)"]]
            Mul_3[["Mul(., .)"]]

            I_xs -->|"FLOAT(batch, cache, 192)"| SimplifiedLayerNormalization_2
            Constant_0 -->|"FLOAT(192)"| SimplifiedLayerNormalization_2
            SimplifiedLayerNormalization_2 -->|"FLOAT(batch, cache, 192)"| Mul_3
            Constant_1 -->|"FLOAT(192)"| Mul_3

            O_a(["a FLOAT()"])
            Mul_3 --> O_a

            class I_xs,I_weights,O_a ioNode
            class Constant_0,Constant_1 constNode
            class SimplifiedLayerNormalization_2,Mul_3 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_xs(["xs FLOAT(batch, cache, 192)"])
            I_weights(["weights FLOAT(192)"])

            SimplifiedLayerNormalization_0[["SimplifiedLayerNormalization(., ., axis=-1)"]]

            I_xs -->|"FLOAT(batch, cache, 192)"| SimplifiedLayerNormalization_0
            I_weights -->|"FLOAT(192)"| SimplifiedLayerNormalization_0

            O_a(["a FLOAT()"])
            SimplifiedLayerNormalization_0 --> O_a

            class I_xs,I_weights,O_a ioNode
            class SimplifiedLayerNormalization_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type, node.domain) != ("SimplifiedLayerNormalization", ""):
            return self.none()
        if len(node.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst_skip = g.get_computed_constant(node.input[1])
        if cst_skip is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        mul_nodes = g.next_nodes(node.output[0])
        if len(mul_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        mul_node = mul_nodes[0]
        if mul_node.op_type != "Mul" or mul_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        index_cst = 1 if mul_node.input[0] == node.output[0] else 0
        if not g.has_shape(mul_node.input[index_cst]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_shape_renamed(node.input[1]) != g.get_shape_renamed(mul_node.input[index_cst]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(mul_node.input[index_cst]):
            # not supported yet
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, mul_node], self.apply)

    def apply(
        self, g: "GraphBuilder", simp_node: NodeProto, mul_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        index_cst = 1 if mul_node.input[0] == simp_node.output[0] else 0
        cst_skip = g.get_computed_constant(simp_node.input[1])
        if cst_skip.min() == cst_skip.max() == 1:
            cst_name = mul_node.input[index_cst]
        else:
            cst2 = g.get_computed_constant(mul_node.input[index_cst])
            if cst2.min() == cst2.max() == 1:
                cst_name = simp_node.input[1]
            else:
                cst2 = g.get_computed_constant(mul_node.input[index_cst])
                new_cst = cst_skip * cst2
                cst_name = g.make_initializer(
                    f"{simp_node.input[1]}__{mul_node.input[index_cst]}",
                    new_cst,
                    source=f"{self.__class__.__name__}.gamma",
                )

        new_node = g.make_node(
            simp_node.op_type,
            [simp_node.input[0], cst_name],
            [mul_node.output[0], *simp_node.output[1:]],
            name=f"{self.__class__.__name__}--{simp_node.name}",
            domain=simp_node.domain,
        )
        if simp_node.attribute:
            new_node.attribute.extend(simp_node.attribute)
        return [new_node]
