import inspect
from typing import List, Optional
import numpy as np
import onnx.numpy_helper as onh
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class TriMatrixPattern(PatternOptimization):
    """
    Replaces a sequence of nodes creating a triangular matrix
    with operator TriMatrix(...).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_shape(["shape INT64(2)"])

            Constant_0[["Constant() -#gt; shape"]]
            Range_1[["Range(0, 1024, 1)"]]
            Add_2[["Add(., [1])"]]
            Reshape_3[["Reshape(., [1024, 1])"]]
            Less_4[["Less(., .)"]]
            Where_5[["Where(., [0.0], .)"]]
            ConstantOfShape_6[["ConstantOfShape(.)"]]

            Range_1 -->|"INT64(NEWDIM_range_0)"| Add_2
            Add_2 -->|"INT64(NEWDIM_range_0)"| Reshape_3
            Range_1 -->|"INT64(NEWDIM_range_0)"| Less_4
            Reshape_3 -->|"INT64(1024, 1)"| Less_4
            Less_4 -->|"BOOL(1024, NEWDIM_range_0)"| Where_5
            ConstantOfShape_6 -->|"FLOAT(1024, 1024)"| Where_5
            Constant_0 -->|"INT64(2)"| ConstantOfShape_6

            O_Y(["Y FLOAT(UNKNOWNDIM, UNKNOWNDIM1)"])
            Where_5 --> O_Y

            class I_shape,O_Y ioNode
            class Constant_0 constNode
            class Range_1,Add_2,Reshape_3,Less_4,Where_5,ConstantOfShape_6 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_shape(["shape INT64(2)"])

            TriMatrix_0[["yaourt.ortops.fused_kernel.cuda.TriMatrix(
            ., [0.0, 0.0, -3.4028235e+38])"]]

            I_shape -->|"INT64(2)"| TriMatrix_0

            O_Y(["Y FLOAT(UNKNOWNDIM, UNKNOWNDIM1)"])
            TriMatrix_0 --> O_Y

            class I_shape,O_Y ioNode
            class TriMatrix_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "Range" or node.domain != "":
            return self.none()

        if (
            len(node.input) != 3
            or not g.is_constant_scalar(node.input[0])
            or not g.is_constant_scalar(node.input[1])
            or not g.is_constant_scalar(node.input[2])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        start, limit, delta = [g.get_constant_scalar(i) for i in node.input]
        if start != 0 or delta != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        types = {n.op_type for n in next_nodes}
        if types != {"Add", "Less"}:
            return self.none(node, inspect.currentframe().f_lineno)

        if next_nodes[0].op_type == "Add":
            add_node, less_node = next_nodes
        else:
            less_node, add_node = next_nodes

        if (
            not g.is_constant_scalar(add_node.input[1])
            or g.get_constant_scalar(add_node.input[1]) != 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        resh_node = g.next_nodes(add_node.output[0])
        if len(resh_node) != 1 or resh_node[0].op_type != "Reshape":
            return self.none(node, inspect.currentframe().f_lineno)

        reshape_node = resh_node[0]

        shape = g.get_computed_constant(reshape_node.input[1])
        if shape.tolist() != [limit, 1]:
            return self.none(node, inspect.currentframe().f_lineno)

        if less_node.input != [node.output[0], reshape_node.output[0]]:
            return self.none(node, inspect.currentframe().f_lineno)

        where_node = g.next_nodes(less_node.output[0])
        if len(where_node) != 1 or where_node[0].op_type != "Where":
            return self.none(node, inspect.currentframe().f_lineno)

        where_node = where_node[0]
        if not g.is_constant_scalar(where_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        cst_node = g.node_before(where_node.input[2])
        if cst_node.op_type != "ConstantOfShape":
            return self.none(node, inspect.currentframe().f_lineno)

        shape = g.get_computed_constant(cst_node.input[0])
        if shape.tolist() != [limit, limit]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [node, add_node, reshape_node, less_node, where_node, cst_node],
            self.apply,
            insert_at=where_node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        range_node: NodeProto,
        add_node: NodeProto,
        reshape_node: NodeProto,
        less_node: NodeProto,
        where_node: NodeProto,
        cst_node: NodeProto,
    ) -> List[NodeProto]:
        cst_upper = onh.to_array(g.get_attribute(cst_node, "value").t)
        dtype = cst_upper.dtype
        cst_lower = np.array([g.get_constant_scalar(where_node.input[1])], dtype=dtype)
        cst_diag = cst_lower
        csts_array = np.hstack([cst_lower, cst_diag, cst_upper]).astype(dtype)
        assert csts_array.shape == (3,), f"Wrong constant array: {csts_array}"

        cst_name = g.make_initializer(
            f"{self.__class__.__name__}--{where_node.name}",
            csts_array,
            source="TriMatrixPattern.apply.cst",
        )
        new_node = g.make_node(
            "TriMatrix",
            [cst_node.input[0], cst_name],
            where_node.output,
            name=f"{self.__class__.__name__}--{where_node.name}",
            domain="yaourt.ortops.fused_kernel.cuda",
        )
        return [new_node]
