import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class GatherConcatPattern(PatternOptimization):
    """
    Simplifies ``Gather(Concat(..., X, ..., axis=0), cst1)`` into
    ``Gather(X, cst2)`` where X is a 1D tensor and cst1, cst2 are 0D or 1D
    integer tensors.

    This applies when:

    - The ``Concat`` axis is 0.
    - Exactly one ``Concat`` input (``X``) is not a constant; all others are
      constants with known sizes.
    - ``cst1`` is a constant 0D or 1D ``INT64`` tensor with non-negative
      values.
    - All indices in ``cst1`` fall within ``X``'s slice of the concatenated
      tensor.

    The adjusted index is ``cst2 = cst1 - offset`` where ``offset`` is the
    sum of the sizes of all constant inputs that precede ``X`` in the
    ``Concat`` input list.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X INT64(n)"])
            I_C1(["C1 INT64(3)"])
            I_C2(["C2 INT64(2)"])
            I_cst1(["cst1 INT64()"])

            Concat_0[["Concat(., ., ., axis=0)"]]
            Gather_1[["Gather(., .)"]]

            I_C1 -->|"INT64(3)"| Concat_0
            I_X -->|"INT64(n)"| Concat_0
            I_C2 -->|"INT64(2)"| Concat_0
            Concat_0 -->|"INT64(3+n+2)"| Gather_1
            I_cst1 -->|"INT64()"| Gather_1

            O_Y(["Y INT64()"])
            Gather_1 --> O_Y

            class I_X,I_C1,I_C2,I_cst1,O_Y ioNode
            class Concat_0,Gather_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X INT64(n)"])
            I_cst2(["cst2 INT64()"])

            Gather_0[["Gather(., .)"]]

            I_X -->|"INT64(n)"| Gather_0
            I_cst2 -->|"INT64()"| Gather_0

            O_Y(["Y INT64()"])
            Gather_0 --> O_Y

            class I_X,I_cst2,O_Y ioNode
            class Gather_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gather" or node.domain != "":
            return self.none()

        # Gather must use axis=0.
        axis = g.get_attribute_with_default(node, "axis", default_value=0)
        if axis != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        # Indices must be a constant 0D or 1D int64/int32 tensor.
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst1 = g.get_computed_constant(node.input[1])
        if cst1 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst1.dtype not in (np.int64, np.int32):
            return self.none(node, inspect.currentframe().f_lineno)
        if cst1.ndim > 1:
            return self.none(node, inspect.currentframe().f_lineno)

        # Data input must come from a Concat node.
        concat_node = g.node_before(node.input[0])
        if concat_node is None or concat_node.op_type != "Concat" or concat_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Concat must use axis=0.
        if g.get_attribute_with_default(concat_node, "axis", default_value=0) != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        # Exactly one Concat input may be non-constant; that is X.
        non_const = [inp for inp in concat_node.input if not g.is_constant(inp)]
        if len(non_const) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        x_name = non_const[0]

        # X must be 1D (rank 1).
        if not g.has_rank(x_name) or g.get_rank(x_name) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        # Locate X in the Concat input list.
        x_idx = next(i for i, inp in enumerate(concat_node.input) if inp == x_name)

        # Compute offset = sum of sizes of all constant inputs before X.
        offset = 0
        for inp in concat_node.input[:x_idx]:
            cst = g.get_computed_constant(inp)
            if cst is None:
                return self.none(node, inspect.currentframe().f_lineno)
            offset += int(cst.shape[0])

        # If constant inputs follow X, verify cst1 doesn't address their region
        # (requires knowing the static size of X).
        after_inputs = list(concat_node.input[x_idx + 1 :])
        x_size: Optional[int] = None
        if after_inputs:
            if not g.has_shape(x_name):
                return self.none(node, inspect.currentframe().f_lineno)
            x_shape = g.get_shape(x_name)
            if x_shape is None or len(x_shape) != 1 or not isinstance(x_shape[0], int):
                return self.none(node, inspect.currentframe().f_lineno)
            x_size = int(x_shape[0])

        # All indices must be non-negative and fall within X's slice.
        indices = np.atleast_1d(cst1).flatten()
        if not np.all(indices >= offset):
            return self.none(node, inspect.currentframe().f_lineno)
        if x_size is not None and not np.all(indices < offset + x_size):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [concat_node, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", concat_node: NodeProto, gather_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        x_name = next(inp for inp in concat_node.input if not g.is_constant(inp))
        x_idx = next(i for i, inp in enumerate(concat_node.input) if inp == x_name)

        # Recompute offset.
        offset = 0
        for inp in concat_node.input[:x_idx]:
            cst = g.get_computed_constant(inp)
            offset += int(cst.shape[0])

        cst1 = g.get_computed_constant(gather_node.input[1])
        cst2 = np.asarray(cst1 - offset, dtype=cst1.dtype)

        new_indices = g.make_initializer("", cst2, source=f"{self.__class__.__name__}.indices")
        new_node = g.make_node(
            "Gather",
            [x_name, new_indices],
            gather_node.output,
            axis=0,
            name=f"{self.__class__.__name__}--{gather_node.name}",
            doc_string=gather_node.doc_string,
        )
        if g.is_used_more_than_once(concat_node.output[0]):
            return [concat_node, new_node]
        return [new_node]


class GatherGatherPattern(PatternOptimization):
    """
    Simplifies two consecutive Gather operations into one.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(n, a, b)"])
            I_cst1(["cst1 INT64(k)"])
            I_cst2(["cst2 INT64()"])

            Constant_0[["Constant() -#gt; cst1"]]
            Constant_1[["Constant() -#gt; cst2"]]
            Gather_2[["Gather(., ., axis=0)"]]
            Gather_3[["Gather(., ., axis=0)"]]

            I_X -->|"FLOAT(n, a, b)"| Gather_2
            Constant_0 -->|"INT64(k)"| Gather_2
            Gather_2 -->|"FLOAT(k, a, b)"| Gather_3
            Constant_1 -->|"INT64()"| Gather_3

            O_Y(["Y FLOAT(a, b)"])
            Gather_3 --> O_Y

            class I_X,I_cst1,I_cst2,O_Y ioNode
            class Constant_0,Constant_1 constNode
            class Gather_2,Gather_3 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(n, a, b)"])
            I_cst3(["cst3 INT64()"])

            Constant_0[["Constant() -#gt; cst3"]]
            Gather_1[["Gather(., ., axis=0)"]]

            I_X -->|"FLOAT(n, a, b)"| Gather_1
            Constant_0 -->|"INT64()"| Gather_1

            O_Y(["Y FLOAT(a, b)"])
            Gather_1 --> O_Y

            class I_X,I_cst3,O_Y ioNode
            class Constant_0 constNode
            class Gather_1 opNode

    The composed index is ``cst3 = cst1[cst2]``. This applies when both
    Gather operations use ``axis=0``, the inner Gather indices ``cst1`` are a
    1-D constant array, and the outer Gather indices ``cst2`` are a constant of
    any shape (scalar or 1-D).
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gather" or node.domain != "":
            return self.none()

        # Outer Gather must use axis=0.
        axis2 = g.get_attribute_with_default(node, "axis", default_value=0)
        if axis2 != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        # Outer indices must be a constant of integer type.
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst2 = g.get_computed_constant(node.input[1])
        if cst2 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst2.dtype not in (np.int64, np.int32):
            return self.none(node, inspect.currentframe().f_lineno)

        # The data input must come from another Gather node.
        inner_node = g.node_before(node.input[0])
        if inner_node is None or inner_node.op_type != "Gather" or inner_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Inner Gather must also use axis=0.
        axis1 = g.get_attribute_with_default(inner_node, "axis", default_value=0)
        if axis1 != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        # Inner indices must be a constant 1-D int64 array.
        if not g.is_constant(inner_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst1 = g.get_computed_constant(inner_node.input[1])
        if cst1 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst1.dtype not in (np.int64, np.int32):
            return self.none(node, inspect.currentframe().f_lineno)
        if cst1.ndim != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [inner_node, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", inner_node: NodeProto, outer_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        cst1 = g.get_computed_constant(inner_node.input[1])
        cst2 = g.get_computed_constant(outer_node.input[1])

        # Compose indices: cst3 = cst1[cst2].
        cst3 = np.asarray(cst1[cst2], dtype=np.int64)

        new_indices = g.make_initializer("", cst3, source=f"{self.__class__.__name__}.indices")
        new_node = g.make_node(
            "Gather",
            [inner_node.input[0], new_indices],
            outer_node.output,
            axis=0,
            name=f"{self.__class__.__name__}--{outer_node.name}",
            doc_string=outer_node.doc_string,
        )
        if g.is_used_more_than_once(inner_node.output[0]):
            return [inner_node, new_node]
        return [new_node]
