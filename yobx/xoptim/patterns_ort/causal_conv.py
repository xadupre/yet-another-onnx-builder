import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class CausalConvWithStatePattern(PatternOptimization):
    """
    Fuses ``Concat + Conv (+ Slice)`` into ``com.microsoft.CausalConvWithState``.

    The operator performs a stateful causal depthwise 1-D convolution and
    replaces the streaming pattern that concatenates a past-state buffer with
    the current input, runs a depthwise Conv, and optionally slices the last
    ``K-1`` frames back out as the next state.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_input(["input FLOAT(N, C, L)"])
            I_weight(["weight FLOAT(C, 1, K)"])
            I_bias(["bias FLOAT(C)"])
            I_state(["past_state FLOAT(N, C, K-1)"])

            Concat_0[["Concat(., ., axis=2)"]]
            Conv_1[["Conv(., ., ., groups=C)"]]
            Slice_2[["Slice(., ., ., [2])"]]

            I_state -->|"FLOAT(N, C, K-1)"| Concat_0
            I_input -->|"FLOAT(N, C, L)"| Concat_0
            Concat_0 -->|"FLOAT(N, C, K-1+L)"| Conv_1
            I_weight -->|"FLOAT(C, 1, K)"| Conv_1
            I_bias -->|"FLOAT(C)"| Conv_1
            Concat_0 -->|"FLOAT(N, C, K-1+L)"| Slice_2

            O_output(["output FLOAT(N, C, L)"])
            Conv_1 --> O_output
            O_state(["present_state FLOAT(N, C, K-1)"])
            Slice_2 --> O_state

            class I_input,I_weight,I_bias,I_state,O_output,O_state ioNode
            class Concat_0,Conv_1,Slice_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_input(["input FLOAT(N, C, L)"])
            I_weight(["weight FLOAT(C, 1, K)"])
            I_bias(["bias FLOAT(C)"])
            I_state(["past_state FLOAT(N, C, K-1)"])

            CausalConvWithState_0[["com.microsoft.CausalConvWithState(., ., ., .)"]]

            I_input -->|"FLOAT(N, C, L)"| CausalConvWithState_0
            I_weight -->|"FLOAT(C, 1, K)"| CausalConvWithState_0
            I_bias -->|"FLOAT(C)"| CausalConvWithState_0
            I_state -->|"FLOAT(N, C, K-1)"| CausalConvWithState_0

            O_output(["output FLOAT(N, C, L)"])
            CausalConvWithState_0 --> O_output
            O_state(["present_state FLOAT(N, C, K-1)"])
            CausalConvWithState_0 --> O_state

            class I_input,I_weight,I_bias,I_state,O_output,O_state ioNode
            class CausalConvWithState_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Conv" or node.domain != "":
            return self.none()

        # Conv must not use auto_pad.
        auto_pad = g.get_attribute_with_default(node, "auto_pad", "NOTSET")
        if auto_pad not in ("NOTSET", b"NOTSET"):
            return self.none(node, inspect.currentframe().f_lineno)

        # Conv must have stride 1 on the sequence dimension (last spatial dim).
        strides = g.get_attribute_with_default(node, "strides", None)
        if strides is not None and any(s != 1 for s in strides):
            return self.none(node, inspect.currentframe().f_lineno)

        # Conv must have dilation 1.
        dilations = g.get_attribute_with_default(node, "dilations", None)
        if dilations is not None and any(d != 1 for d in dilations):
            return self.none(node, inspect.currentframe().f_lineno)

        # Conv must have no padding (causal context already provided by the Concat).
        pads = g.get_attribute_with_default(node, "pads", None)
        if pads is not None and any(p != 0 for p in pads):
            return self.none(node, inspect.currentframe().f_lineno)

        # The Conv's data input must come from a Concat node.
        concat_node = g.node_before(node.input[0])
        if concat_node is None or concat_node.op_type != "Concat" or concat_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Concat must join exactly 2 tensors (past_state and current input).
        if len(concat_node.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        # Concat axis must be 2 (the sequence/time dimension for 1-D convolution).
        concat_axis = g.get_attribute_with_default(concat_node, "axis", 1)
        if concat_axis not in (2, -1):
            return self.none(node, inspect.currentframe().f_lineno)

        # Conv must be a depthwise convolution: groups == number of input channels.
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        weight_shape = g.get_shape(node.input[1])
        if len(weight_shape) != 3:
            # Only 1-D depthwise case is supported.
            return self.none(node, inspect.currentframe().f_lineno)
        out_channels = weight_shape[0]
        groups = g.get_attribute_with_default(node, "group", 1)
        if groups != out_channels:
            return self.none(node, inspect.currentframe().f_lineno)
        # For depthwise conv the per-group input channel count must be 1.
        if weight_shape[1] != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        # The Concat output may be used by both Conv and an optional Slice.
        concat_consumers = g.next_nodes(concat_node.output[0])
        if len(concat_consumers) not in (1, 2):
            return self.none(node, inspect.currentframe().f_lineno)

        # Look for an optional Slice that extracts the present_state.
        slice_node: Optional[NodeProto] = None
        for consumer in concat_consumers:
            if consumer is node:
                continue
            if consumer.op_type == "Slice" and consumer.domain == "":
                slice_node = consumer
            else:
                # Unknown extra consumer – bail out.
                return self.none(node, inspect.currentframe().f_lineno)

        # Validate the Slice when present.
        if slice_node is not None:
            # Slice must consume the Concat output as its data input.
            if slice_node.input[0] != concat_node.output[0]:
                return self.none(node, inspect.currentframe().f_lineno)

            # Slice must have constant axes input equal to [2].
            if len(slice_node.input) < 4 or not g.is_constant(slice_node.input[3]):
                return self.none(node, inspect.currentframe().f_lineno)
            axes_cst = g.get_computed_constant(slice_node.input[3])
            if axes_cst is None:
                return self.none(node, inspect.currentframe().f_lineno)
            axes = axes_cst.flatten().tolist()
            if axes not in ([2], [-1]):
                return self.none(node, inspect.currentframe().f_lineno)

        nodes = [concat_node, node]
        if slice_node is not None:
            nodes.append(slice_node)

        last_node = slice_node if slice_node is not None else node
        return MatchResult(self, nodes, self.apply, insert_at=last_node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        concat_node: NodeProto,
        conv_node: NodeProto,
        slice_node: Optional[NodeProto] = None,
    ) -> List[NodeProto]:
        # past_state is the first input to Concat, current input is the second.
        past_state = concat_node.input[0]
        x = concat_node.input[1]
        weight = conv_node.input[1]
        bias = conv_node.input[2] if len(conv_node.input) > 2 else ""

        # Determine output names.
        conv_output = conv_node.output[0]
        state_output = slice_node.output[0] if slice_node is not None else ""

        fused_outputs = [conv_output]
        if state_output:
            fused_outputs.append(state_output)

        fused_node = g.make_node(
            "CausalConvWithState",
            [x, weight, bias, past_state],
            fused_outputs,
            domain="com.microsoft",
            ndim=1,
            name=f"{self.__class__.__name__}--{conv_node.name}",
        )
        return [fused_node]
