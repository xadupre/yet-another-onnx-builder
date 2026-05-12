import inspect
from typing import List, Optional

from onnx import NodeProto, TensorProto

from ..patterns_api import MatchResult, PatternOptimization

# Indices of the integer inputs of com.microsoft.GreedySearch.
# input[0]: input_ids     (batch_size, sequence_length) INT32
# input[1]: max_length    (1,) INT32
# input[2]: min_length    (1,) INT32  -- optional
# input[3]: repetition_penalty (1,) FLOAT -- skip, not an integer input
# input[4]: vocab_mask    (vocab_size,) INT32  -- optional
# input[5]: prefix_vocab_mask (batch_size, vocab_size) INT32  -- optional
# input[6]: attention_mask (batch_size, sequence_length) INT32  -- optional
_GREEDY_SEARCH_INTEGER_INPUT_INDICES = (0, 1, 2, 4, 5, 6)


class GreedySearchPattern(PatternOptimization):
    """
    Ensures ``com.microsoft.GreedySearch`` receives INT32 integer inputs.

    The ORT contrib operator ``GreedySearch`` requires all integer tensors
    (``input_ids``, ``max_length``, ``min_length``, ``vocab_mask``,
    ``prefix_vocab_mask``, and ``attention_mask``) to be of type INT32.
    PyTorch typically produces INT64 tensors, so without this pattern the
    node would fail at runtime.

    This pattern matches any ``com.microsoft.GreedySearch`` node that has at
    least one integer input with dtype INT64 and inserts ``Cast(INT64→INT32)``
    nodes for every such input.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_input_ids(["input_ids INT64(batch, seq)"])
            I_max_length(["max_length INT64(1)"])

            GreedySearch_0[["com.microsoft.GreedySearch(.decoder, .)"]]

            I_input_ids -->|"INT64(batch, seq)"| GreedySearch_0
            I_max_length -->|"INT64(1)"| GreedySearch_0

            O_sequences(["sequences INT32(batch, max_seq)"])
            GreedySearch_0 --> O_sequences

            class I_input_ids,I_max_length,O_sequences ioNode
            class GreedySearch_0 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_input_ids(["input_ids INT64(batch, seq)"])
            I_max_length(["max_length INT64(1)"])

            Cast_0[["Cast(., to=INT32)"]]
            Cast_1[["Cast(., to=INT32)"]]
            GreedySearch_2[["com.microsoft.GreedySearch(.decoder, .)"]]

            I_input_ids -->|"INT64(batch, seq)"| Cast_0
            I_max_length -->|"INT64(1)"| Cast_1
            Cast_0 -->|"INT32(batch, seq)"| GreedySearch_2
            Cast_1 -->|"INT32(1)"| GreedySearch_2

            O_sequences(["sequences INT32(batch, max_seq)"])
            GreedySearch_2 --> O_sequences

            class I_input_ids,I_max_length,O_sequences ioNode
            class Cast_0,Cast_1,GreedySearch_2 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "GreedySearch" or node.domain != "com.microsoft":
            return self.none()

        needs_cast = False
        for idx in _GREEDY_SEARCH_INTEGER_INPUT_INDICES:
            if idx < len(node.input) and node.input[idx]:
                if (
                    g.has_type(node.input[idx])
                    and g.get_type(node.input[idx]) == TensorProto.INT64
                ):
                    needs_cast = True
                    break

        if not needs_cast:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        new_inputs = list(node.input)
        cast_nodes: List[NodeProto] = []

        for idx in _GREEDY_SEARCH_INTEGER_INPUT_INDICES:
            if idx >= len(new_inputs) or not new_inputs[idx]:
                continue
            inp = new_inputs[idx]
            if not g.has_type(inp) or g.get_type(inp) != TensorProto.INT64:
                continue

            cast_out = g.unique_name(f"{self.__class__.__name__}--cast{idx}--{inp}")
            cast_nodes.append(
                g.make_node(
                    "Cast",
                    [inp],
                    [cast_out],
                    to=TensorProto.INT32,
                    name=f"{self.__class__.__name__}--Cast{idx}--{node.name}",
                )
            )
            new_inputs[idx] = cast_out

        new_node = g.make_node(
            "GreedySearch",
            new_inputs,
            node.output,
            domain="com.microsoft",
            name=f"{self.__class__.__name__}--{node.name}",
        )
        new_node.attribute.extend(node.attribute)
        return [*cast_nodes, new_node]
