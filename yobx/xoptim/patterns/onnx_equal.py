import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class UnsqueezeEqualPattern(PatternOptimization):
    """
    Replaces the sequence R -> Equal -> Unsqueeze, R -> Unsqueeze,
    into R -> Unsqueeze -> Equal.

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
                    oh.make_node('Constant', [], ['axis'], value=onh.from_array(np.array([1], dtype=np.int64), name='value')),
                    oh.make_node('Constant', [], ['m_one'], value=onh.from_array(np.array([-1.0], dtype=np.float32), name='value')),
                    oh.make_node('Unsqueeze', ['X', 'axis'], ['Y']),
                    oh.make_node('Equal', ['X', 'm_one'], ['xe']),
                    oh.make_node('Unsqueeze', ['xe', 'axis'], ['Z']),
                ],
                'pattern',
                [
                    oh.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, ('a', 1, 'b')),
                    oh.make_tensor_value_info('m_one', onnx.TensorProto.FLOAT, (1,)),
                    oh.make_tensor_value_info('X', onnx.TensorProto.FLOAT, ('a', 'b')),
                    oh.make_tensor_value_info('axis', onnx.TensorProto.INT64, (1,)),
                ],
                [
                    oh.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, ('a', 1, 'b')),
                    oh.make_tensor_value_info('Z', onnx.TensorProto.BOOL, ('a', 1, 'b')),
                ],
            ),
            functions=[],
            opset_imports=[oh.make_opsetid('', 26)],
        )

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from yobx.doc import to_dot
        import onnx
        import onnx.helper as oh

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node('Unsqueeze', ['X', 'axis'], ['Y']),
                    oh.make_node('Equal', ['Y', 'm_one'], ['Z']),
                ],
                'pattern',
                [
                    oh.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, ('a', 1, 'b')),
                    oh.make_tensor_value_info('m_one', onnx.TensorProto.FLOAT, (1,)),
                    oh.make_tensor_value_info('X', onnx.TensorProto.FLOAT, ('a', 'b')),
                    oh.make_tensor_value_info('axis', onnx.TensorProto.INT64, (1,)),
                ],
                [
                    oh.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, ('a', 1, 'b')),
                    oh.make_tensor_value_info('Z', onnx.TensorProto.BOOL, ('a', 1, 'b')),
                ],
            ),
            functions=[],
            opset_imports=[oh.make_opsetid('', 26)],
        )

        print("DOT-SECTION", to_dot(model))
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Equal" or node.domain != "":
            return self.none()
        if not g.is_constant_scalar(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        after = g.next_nodes(node.output[0])
        if len(after) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        next_path = g.next_nodes(node.input[0])
        if len(next_path) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        if next_path[0].op_type == node.op_type and next_path[1].op_type == "Unsqueeze":
            if next_path[1].input[1] != after[0].input[1]:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [next_path[1], node, after[0]], self.apply)
        if next_path[1].op_type == node.op_type and next_path[0].op_type == "Unsqueeze":
            if next_path[0].input[1] != after[0].input[1]:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [next_path[0], node, after[0]], self.apply)
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_unsqueeze: NodeProto,
        node_equal: NodeProto,
        node_equal_unsqueeze: NodeProto,
    ) -> List[NodeProto]:
        return [
            node_unsqueeze,
            g.make_node(
                node_equal.op_type,
                [node_unsqueeze.output[0], node_equal.input[1]],
                [node_equal_unsqueeze.output[0]],
                domain=node_equal.domain,
                name=f"{self.__class__.__name__}--{node_equal.name}",
            ),
        ]
