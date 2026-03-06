import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ConstantToInitializerPattern(PatternOptimization):
    """
    Replaces a node Constant by an initializer and a node Identity.

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from yobx.doc import to_dot, make_pattern_model
        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        inputs = []
        outputs = []
        nodes = []
        initializers = []
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["cst"],
                value=onh.from_array(np.array([1.0, 2.0], dtype=np.float32), name="value"),
            )
        )
        outputs.append(oh.make_tensor_value_info("cst", onnx.TensorProto.FLOAT, shape=(2,)))
        model = make_pattern_model(nodes, inputs, outputs, initializers)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from yobx.doc import to_dot, make_pattern_model
        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        inputs = []
        outputs = []
        nodes = []
        initializers = []
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["cst_cst2init"],
                value=onh.from_array(np.array([1.0, 2.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(oh.make_node("Identity", ["cst_cst2init"], ["cst"]))
        outputs.append(oh.make_tensor_value_info("cst", onnx.TensorProto.FLOAT, shape=(2,)))
        model = make_pattern_model(nodes, inputs, outputs, initializers)

        print("DOT-SECTION", to_dot(model))
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Constant" or node.domain != "":
            return self.none()
        if g.do_not_turn_constant_initializers_maybe_because_of_showing(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        cst = g.get_computed_constant(node.output[0])
        assert (
            cst is not None
        ), f"Node {g.pretty_node(node)} is a constant, it must be possible to evaluate it."
        # if not g.has_exact_same_constant_in_context(node.output[0]):
        init = g.make_initializer(f"{node.output[0]}_cst2init", cst)
        return [
            g.make_node(
                "Identity", [init], node.output, name=f"{self.__class__.__name__}--{node.name}"
            )
        ]
