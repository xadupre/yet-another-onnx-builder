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

        from yobx.doc import to_dot
        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node('Constant', [], ['cst'],
                                 value=onh.from_array(np.array([1.0, 2.0], dtype=np.float32),
                                 name='value')),
                ],
                'pattern',
                [
                ],
                [
                    oh.make_tensor_value_info('cst', onnx.TensorProto.FLOAT, (2,)),
                ],
            ),
            functions=[],
            opset_imports=[oh.make_opsetid('', 18)],
        )

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from yobx.doc import to_dot
        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node('Constant', [], ['cst_cst2init'],
                                 value=onh.from_array(np.array([1.0, 2.0], dtype=np.float32),
                                 name='value')),
                    oh.make_node('Identity', ['cst_cst2init'], ['cst']),
                ],
                'pattern',
                [
                ],
                [
                    oh.make_tensor_value_info('cst', onnx.TensorProto.FLOAT, (2,)),
                ],
            ),
            functions=[],
            opset_imports=[oh.make_opsetid('', 18)],
        )

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

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
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
