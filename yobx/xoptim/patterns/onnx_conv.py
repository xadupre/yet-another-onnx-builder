import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto, helper as oh
from ..patterns_api import MatchResult, PatternOptimization


class ConvBiasNullPattern(PatternOptimization):
    """
    Checks that a Conv has a null bias.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(512, 3, 64, 64)"])
            I_W(["W FLOAT(64, 3, 4, 4)"])
            i_B2["B2 FLOAT(64)"]

            Conv_0[["Conv(., ., .)"]]

            I_X -->|"FLOAT(512, 3, 64, 64)"| Conv_0
            I_W -->|"FLOAT(64, 3, 4, 4)"| Conv_0
            i_B2 -->|"FLOAT(64)"| Conv_0

            O_Y(["Y FLOAT(512, 64, 32, 32)"])
            Conv_0 --> O_Y

            class I_X,I_W,O_Y ioNode
            class i_B2 initNode
            class Conv_0 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(512, 3, 64, 64)"])
            I_W(["W FLOAT(64, 3, 4, 4)"])

            Conv_0[["Conv(., .)"]]

            I_X -->|"FLOAT(512, 3, 64, 64)"| Conv_0
            I_W -->|"FLOAT(64, 3, 4, 4)"| Conv_0

            O_Y(["Y FLOAT(512, 64, 32, 32)"])
            Conv_0 --> O_Y

            class I_X,I_W,O_Y ioNode
            class Conv_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Conv" or node.domain != "":
            return self.none()
        if len(node.input) < 3:
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)

        cst = g.get_computed_constant(node.input[2])
        if cst is None or cst.min() != 0 or cst.max() != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        new_node = g.make_node(
            "Conv",
            node.input[:2],
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        new_node.attribute.extend(node.attribute)
        return [new_node]


class PadConvPattern(PatternOptimization):
    """
    Fuses a Pad node followed by a Conv node into a single Conv node
    with the padding folded into the ``pads`` attribute.

    The fusion is valid when:

    * The Pad mode is ``constant`` (default).
    * The constant padding value is ``0`` (default).
    * The padding is applied only to spatial dimensions (batch and channel
      dimensions must have zero padding).
    * The Conv does not already use ``auto_pad`` (other than ``NOTSET``).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(1, 3, 6, 6)"])
            I_W(["W FLOAT(8, 3, 3, 3)"])
            i_pads["pads INT64(8)"]

            Pad_0[["Pad(., .)"]]
            Conv_1[["Conv(., .)"]]

            I_X -->|"FLOAT(1, 3, 6, 6)"| Pad_0
            i_pads -->|"INT64(8)"| Pad_0
            Pad_0 -->|"FLOAT(1, 3, 8, 8)"| Conv_1
            I_W -->|"FLOAT(8, 3, 3, 3)"| Conv_1

            O_Y(["Y FLOAT(1, 8, 6, 6)"])
            Conv_1 --> O_Y

            class I_X,I_W,O_Y ioNode
            class i_pads initNode
            class Pad_0,Conv_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(1, 3, 6, 6)"])
            I_W(["W FLOAT(8, 3, 3, 3)"])

            Conv_0[["Conv(., .)"]]

            I_X -->|"FLOAT(1, 3, 6, 6)"| Conv_0
            I_W -->|"FLOAT(8, 3, 3, 3)"| Conv_0

            O_Y(["Y FLOAT(1, 8, 6, 6)"])
            Conv_0 --> O_Y

            class I_X,I_W,O_Y ioNode
            class Conv_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Conv" or node.domain != "":
            return self.none()

        # Conv must not use auto_pad (other than NOTSET).
        auto_pad = g.get_attribute_with_default(node, "auto_pad", "NOTSET")
        if auto_pad not in ("NOTSET", b"NOTSET"):
            return self.none(node, inspect.currentframe().f_lineno)

        # First input of Conv must come from a Pad node.
        pad_node = g.node_before(node.input[0])
        if pad_node is None or pad_node.op_type != "Pad" or pad_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # The Pad output must be consumed only by this Conv.
        if len(g.next_nodes(node.input[0])) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        # Pad mode must be "constant" (default).
        mode = g.get_attribute_with_default(pad_node, "mode", "constant")
        if mode not in ("constant", b"constant"):
            return self.none(node, inspect.currentframe().f_lineno)

        # pads input must be a known constant.
        if len(pad_node.input) < 2 or not g.is_constant(pad_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        pads_cst = g.get_computed_constant(pad_node.input[1])
        if pads_cst is None:
            return self.none(node, inspect.currentframe().f_lineno)
        pads_values = pads_cst.flatten().tolist()

        # constant_value must be 0 (default or explicit).
        if len(pad_node.input) >= 3 and pad_node.input[2] != "":
            if not g.is_constant(pad_node.input[2]):
                return self.none(node, inspect.currentframe().f_lineno)
            cv = g.get_computed_constant(pad_node.input[2])
            if cv is None or float(np.array(cv).flat[0]) != 0.0:
                return self.none(node, inspect.currentframe().f_lineno)

        # axes input (opset 18+): if present it must be a constant.
        axes = None
        if len(pad_node.input) >= 4 and pad_node.input[3] != "":
            if not g.is_constant(pad_node.input[3]):
                return self.none(node, inspect.currentframe().f_lineno)
            axes_cst = g.get_computed_constant(pad_node.input[3])
            if axes_cst is None:
                return self.none(node, inspect.currentframe().f_lineno)
            axes = [int(a) for a in axes_cst.flatten().tolist()]

        # Determine spatial padding from the pads tensor.
        # pads format: [begin_dim0, begin_dim1, ..., end_dim0, end_dim1, ...]
        ndim = len(pads_values) // 2
        if axes is not None:
            # Reconstruct a full pads array from the axes-based pads.
            full_pads = [0] * (2 * ndim)
            for idx, ax in enumerate(axes):
                ax = ax if ax >= 0 else ax + ndim
                full_pads[ax] = pads_values[idx]
                full_pads[ax + ndim] = pads_values[idx + len(axes)]
            pads_values = full_pads

        # Batch and channel dimensions (0 and 1) must have no padding.
        if pads_values[0] != 0 or pads_values[1] != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        if pads_values[ndim] != 0 or pads_values[ndim + 1] != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        # Spatial pads from Pad node: [h_begin, w_begin, ..., h_end, w_end, ...]
        spatial_pads_begin = pads_values[2:ndim]
        spatial_pads_end = pads_values[ndim + 2:]

        # All spatial pads must be non-negative.
        if any(p < 0 for p in spatial_pads_begin + spatial_pads_end):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [pad_node, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        pad_node: NodeProto,
        conv_node: NodeProto,
    ) -> List[NodeProto]:
        # Retrieve the pads from the Pad node.
        pads_cst = g.get_computed_constant(pad_node.input[1])
        pads_values = pads_cst.flatten().tolist()
        ndim = len(pads_values) // 2

        axes = None
        if len(pad_node.input) >= 4 and pad_node.input[3] != "":
            axes_cst = g.get_computed_constant(pad_node.input[3])
            axes = [int(a) for a in axes_cst.flatten().tolist()]

        if axes is not None:
            full_pads = [0] * (2 * ndim)
            for idx, ax in enumerate(axes):
                ax = ax if ax >= 0 else ax + ndim
                full_pads[ax] = pads_values[idx]
                full_pads[ax + ndim] = pads_values[idx + len(axes)]
            pads_values = full_pads

        spatial_pads_begin = list(map(int, pads_values[2:ndim]))
        spatial_pads_end = list(map(int, pads_values[ndim + 2:]))

        # Retrieve existing Conv pads attribute (defaults to all-zeros).
        n_spatial = ndim - 2
        existing_pads_attr = g.get_attribute(conv_node, "pads", exc=False)
        if existing_pads_attr is not None:
            existing_pads = list(existing_pads_attr.ints)
        else:
            existing_pads = [0] * (2 * n_spatial)

        # Merge: Conv pads = existing pads + Pad spatial pads.
        # Pad existing_pads to the expected length if necessary.
        if len(existing_pads) < 2 * n_spatial:
            existing_pads = existing_pads + [0] * (2 * n_spatial - len(existing_pads))
        new_pads = [
            existing_pads[i] + spatial_pads_begin[i] for i in range(n_spatial)
        ] + [
            existing_pads[n_spatial + i] + spatial_pads_end[i] for i in range(n_spatial)
        ]

        new_node = g.make_node(
            "Conv",
            [pad_node.input[0]] + list(conv_node.input[1:]),
            conv_node.output,
            name=f"{self.__class__.__name__}--{conv_node.name}",
            doc_string=conv_node.doc_string,
        )
        # Copy all attributes from the original Conv, replacing/adding pads.
        for att in conv_node.attribute:
            if att.name != "pads":
                new_node.attribute.append(att)

        new_node.attribute.append(oh.make_attribute("pads", new_pads))
        return [new_node]
