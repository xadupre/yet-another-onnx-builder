import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...helpers.onnx_helper import make_idn
from ..patterns_api import MatchResult, PatternOptimization


class SlicesSplitPattern(PatternOptimization):
    """
    Merges multiple parallel slices into a split.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_transpose_1(["transpose_1 FLOAT16(2, 2, 1024, 512)"])

            Slice_0[["Slice(., [0], [256], [3])"]]
            Slice_1[["Slice(., [256], [9223372036854775807], [3])"]]

            I_transpose_1 -->|"FLOAT16(2, 2, 1024, 512)"| Slice_0
            I_transpose_1 -->|"FLOAT16(2, 2, 1024, 512)"| Slice_1

            O_slice_11(["slice_11 FLOAT16(2, 2, 1024, 256)"])
            Slice_0 --> O_slice_11
            O_slice_12(["slice_12 FLOAT16(2, 2, 1024, 256)"])
            Slice_1 --> O_slice_12

            class I_transpose_1,O_slice_11,O_slice_12 ioNode
            class Slice_0,Slice_1 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_transpose_1(["transpose_1 FLOAT16(2, 2, 1024, 512)"])

            Split_0[["Split(., [256, 256], axis=3)"]]

            I_transpose_1 -->|"FLOAT16(2, 2, 1024, 512)"| Split_0

            O_slice_11(["slice_11 FLOAT16(2, 2, 1024, 256)"])
            Split_0 --> O_slice_11
            O_slice_12(["slice_12 FLOAT16(2, 2, 1024, 256)"])
            Split_0 --> O_slice_12

            class I_transpose_1,O_slice_11,O_slice_12 ioNode
            class Split_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Slice" or node.domain != "":
            return self.none()

        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        users = [
            op for op in g.next_nodes(node.input[0]) if op.op_type == "Slice" and op.domain == ""
        ]
        if len(users) <= 1:
            return self.none(node, inspect.currentframe().f_lineno)

        for user in users:
            if len(user.input) == 4:
                continue
            if len(user.input) == 5:
                if not g.is_constant_scalar(user.input[-1]):
                    return self.none(node, inspect.currentframe().f_lineno)
                scalar = g.get_constant_scalar(user.input[-1])
                if scalar != 1:
                    return self.none(node, inspect.currentframe().f_lineno)
                continue
            return self.none(node, inspect.currentframe().f_lineno)

        # axis
        if all(len(op.input) == 2 for op in users):
            axis = 0
        else:
            axes = [op.input[3] for op in users]
            if any(not g.is_constant_scalar(a) for a in axes):
                return self.none(node, inspect.currentframe().f_lineno)

            csts = [g.get_constant_scalar(a) for a in axes]
            if len(set(csts)) != 1:
                return self.none(node, inspect.currentframe().f_lineno)

            axis = csts[0]

        shape = g.get_shape(node.input[0])
        dim = shape[axis]
        if not isinstance(dim, int):
            return self.none(node, inspect.currentframe().f_lineno)

        # starts, ends
        starts = [op.input[1] for op in users]
        ends = [op.input[2] for op in users]

        if not g.is_constant_scalar(starts[0], 0):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(ends[-1]):
            return self.none(node, inspect.currentframe().f_lineno)
        last = g.get_constant_scalar(ends[-1])
        if last not in (dim, 9223372036854775807):
            # 9223372036854775807 is what torch uses to specify the end
            return self.none(node, inspect.currentframe().f_lineno)

        if any(not g.is_constant(i) for i in starts) or any(not g.is_constant(i) for i in ends):
            # no constants
            return self.none(node, inspect.currentframe().f_lineno)

        cst_starts = [None for a in starts]
        cst_ends = [None for a in ends]
        for i in range(len(starts) - 1):
            if ends[i] == starts[i + 1]:
                continue
            end = cst_ends[i] or g.get_computed_constant(ends[i])
            start = cst_starts[i + 1] or g.get_computed_constant(starts[i + 1])
            if all(end == start):
                cst_ends[i] = end
                cst_starts[i + 1] = start
                continue
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, users, self.apply)

    def apply(self, g: "GraphBuilder", *nodes: NodeProto) -> List[NodeProto]:  # noqa: F821
        # nodes are all slices

        starts = [op.input[1] for op in nodes]
        ends = [op.input[2] for op in nodes]
        cst_starts = [g.get_constant_scalar(a) for a in starts]
        cst_ends = [g.get_constant_scalar(a) for a in ends]
        axis = g.get_constant_scalar(nodes[0].input[3])
        if cst_ends[-1] == 9223372036854775807:
            # 9223372036854775807 is what torch uses to specify the end
            shape = g.get_shape(nodes[0].input[0])
            cst_ends[-1] = shape[axis]

        n_els = []
        total_int = None
        for i in range(len(starts)):
            if (cst_ends[i] < 0 and cst_starts[i] >= 0) or (
                cst_ends[i] >= 0 and cst_starts[i] < 0
            ):
                if total_int is None:
                    if g.has_shape(nodes[0].input[0]):
                        shape = g.get_shape(nodes[0].input[0])
                        if isinstance(shape[axis], int):
                            total_int = shape[axis]
                assert total_int is not None, "should not be possible"
                delta = (
                    (cst_ends[i] + total_int - cst_starts[i])
                    if cst_ends[i] < 0
                    else (cst_ends[i] - cst_starts[i] - total_int)
                )
            else:
                delta = cst_ends[i] - cst_starts[i]
            assert delta >= 0, f"{delta=} < 0, {cst_starts[i]=}, {cst_ends[i]=}, {total_int=}"
            n_els.append(delta)

        splits = g.make_initializer(
            "", np.array(n_els, dtype=np.int64), source="SlicesSplitPattern.apply.splits"
        )
        outputs = [op.output[0] for op in nodes]
        node = g.make_node(
            "Split",
            [nodes[0].input[0], splits],
            outputs,
            axis=axis,
            name=f"{self.__class__.__name__}--{nodes[0].name}",
        )
        return [node]


class GathersSplitPattern(PatternOptimization):
    """
    Merges multiple parallel gather into a split followed by unsqueeze.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, 2)"])

            Gather_0[["Gather(., 0, axis=1)"]]
            Gather_1[["Gather(., 1, axis=1)"]]

            I_X -->|"FLOAT(a, 2)"| Gather_0
            I_X -->|"FLOAT(a, 2)"| Gather_1

            O_x2(["x2 FLOAT(a)"])
            Gather_1 --> O_x2
            O_x1(["x1 FLOAT(a)"])
            Gather_0 --> O_x1

            class I_X,O_x2,O_x1 ioNode
            class Gather_0,Gather_1 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, 2)"])

            Split_0[["Split(., axis=1)"]]
            Squeeze_1[["Squeeze(., [1])"]]
            Squeeze_2[["Squeeze(., [1])"]]

            I_X -->|"FLOAT(a, 2)"| Split_0
            Split_0 -->|"FLOAT(a, 1)"| Squeeze_1
            Split_0 -->|"FLOAT(a, 1)"| Squeeze_2

            O_x2(["x2 FLOAT(a)"])
            Squeeze_2 --> O_x2
            O_x1(["x1 FLOAT(a)"])
            Squeeze_1 --> O_x1

            class I_X,O_x2,O_x1 ioNode
            class Split_0,Squeeze_1,Squeeze_2 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gather" or node.domain != "":
            return self.none()

        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        users = [
            op for op in g.next_nodes(node.input[0]) if op.op_type == "Gather" and op.domain == ""
        ]
        if len(users) <= 1:
            return self.none(node, inspect.currentframe().f_lineno)

        axis = None
        csts = set()
        rank = None
        keep_users = []
        for user in users:
            if len(user.input) != 2:
                continue
            a = g.get_attribute_with_default(user, "axis", default_value=0)
            assert a is not None, f"user={user}"
            if axis is not None and a != axis:
                return self.none(node, inspect.currentframe().f_lineno)
            axis = a
            if not g.is_constant_scalar(user.input[1]):
                continue
            cst = g.get_constant_scalar(user.input[1])
            if cst is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if cst in csts:
                return self.none(node, inspect.currentframe().f_lineno)
            rk = g.get_rank(user.input[1])
            if rank is not None and rk != rank:
                return self.none(node, inspect.currentframe().f_lineno)
            rank = rk
            csts.add(cst)
            keep_users.append(user)

        users = keep_users
        sorted_indices = sorted(csts)
        if sorted_indices != list(range(len(csts))):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(node.input[0])
        if axis < 0:
            axis += len(shape)
        if axis >= len(shape):
            return self.none(node, inspect.currentframe().f_lineno)
        if not isinstance(shape[axis], int):
            return self.none(node, inspect.currentframe().f_lineno)
        if shape[axis] != len(sorted_indices):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, users, self.apply)

    def apply(self, g: "GraphBuilder", *gather_nodes: NodeProto) -> List[NodeProto]:  # noqa: F821
        # nodes are all slices

        axis = g.get_attribute_with_default(gather_nodes[0], "axis", default_value=0)
        outputs = [None for u in gather_nodes]
        rank = g.get_rank(gather_nodes[0].input[1])
        post_nodes = []
        if rank == 0:
            axis_init = g.make_initializer(
                "", np.array([axis], dtype=np.int64), source=f"{self.__class__.__name__}.axes"
            )
        for user in gather_nodes:
            cst = g.get_constant_scalar(user.input[1])
            if rank == 1:
                outputs[cst] = user.output[0]
            else:
                name = g.unique_name(f"{self.__class__.__name__}--{user.output[0]}")
                post_nodes.append(
                    g.make_node(
                        "Squeeze",
                        [name, axis_init],
                        [user.output[0]],
                        name=f"{self.__class__.__name__}--{user.name}",
                    )
                )
                outputs[cst] = name

        node = g.make_node(
            "Split",
            [gather_nodes[0].input[0]],
            outputs,
            axis=axis,
            num_outputs=len(outputs),
            name=f"{self.__class__.__name__}--{gather_nodes[0].name}",
        )
        return [node, *post_nodes]


class SplitConcatPattern(PatternOptimization):
    """
    Replaces Split + Concat into identity if this is equivalent.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])

            Split_0[["Split(., axis=-1)"]]
            Concat_1[["Concat(., ., axis=-1)"]]

            I_X -->|"FLOAT(a, b)"| Split_0
            Split_0 -->|"FLOAT(a, CeilToInt(b,2))"| Concat_1
            Split_0 -->|"FLOAT(a, b-CeilToInt(b,2))"| Concat_1

            O_Y(["Y FLOAT(a, b)"])
            Concat_1 --> O_Y

            class I_X,O_Y ioNode
            class Split_0,Concat_1 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])

            Identity_0[["Identity(.)"]]

            I_X -->|"FLOAT(a, b)"| Identity_0

            O_Y(["Y FLOAT(a, b)"])
            Identity_0 --> O_Y

            class I_X,O_Y ioNode
            class Identity_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Split" or node.domain != "":
            return self.none()

        only_id = None
        only_node = None
        for o in node.output:
            n = g.next_nodes(o)
            if len(n) != 1:
                return self.none(node, inspect.currentframe().f_lineno)
            i = make_idn(n[0])
            if only_id is None:
                only_id = i
                only_node = n[0]
            elif i != only_id:
                return self.none(node, inspect.currentframe().f_lineno)

        if only_node.op_type != "Concat" or only_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        axis_split = g.get_attribute(node, "axis").i
        axis_concat = g.get_attribute(only_node, "axis").i
        if axis_split < 0 and axis_concat >= 0:
            axis_split += g.get_rank(node.input[0])
        if axis_concat < 0 and axis_split >= 0:
            axis_concat += g.get_rank(node.input[0])
        if axis_split != axis_concat:
            return self.none(node, inspect.currentframe().f_lineno)
        if node.output != only_node.input:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, only_node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", split_node: NodeProto, concat_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "Identity",
                split_node.input,
                concat_node.output,
                name=f"{self.__class__.__name__}--{split_node.name}",
            )
        ]
