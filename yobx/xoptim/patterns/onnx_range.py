import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SwapRangeAddScalarPattern(PatternOptimization):
    """
    Swap Range + Add when a scalar is added.

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
                    oh.make_node('Constant', [], ['one'], value=onh.from_array(np.array(1, dtype=np.int64), name='value')),
                    oh.make_node('Range', ['START', 'END', 'one'], ['arange']),
                    oh.make_node('Add', ['arange', 'PLUS'], ['Y']),
                ],
                'pattern',
                [
                    oh.make_tensor_value_info('END', onnx.TensorProto.INT64, ()),
                    oh.make_tensor_value_info('PLUS', onnx.TensorProto.INT64, (1,)),
                    oh.make_tensor_value_info('one', onnx.TensorProto.INT64, ()),
                    oh.make_tensor_value_info('START', onnx.TensorProto.INT64, ()),
                ],
                [
                    oh.make_tensor_value_info('Y', onnx.TensorProto.INT64, ('NEWDIM_range',)),
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
        import onnx
        import onnx.helper as oh

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node('Squeeze', ['PLUS'], ['SwapRangeAddScalarPattern--PLUS']),
                    oh.make_node('Add', ['END', 'SwapRangeAddScalarPattern--PLUS'], ['SwapRangeAddScalarPattern--END']),
                    oh.make_node('Add', ['START', 'SwapRangeAddScalarPattern--PLUS'], ['SwapRangeAddScalarPattern--START']),
                    oh.make_node('Range', ['SwapRangeAddScalarPattern--START', 'SwapRangeAddScalarPattern--END', 'one'], ['Y']),
                ],
                'pattern',
                [
                    oh.make_tensor_value_info('END', onnx.TensorProto.INT64, ()),
                    oh.make_tensor_value_info('PLUS', onnx.TensorProto.INT64, (1,)),
                    oh.make_tensor_value_info('one', onnx.TensorProto.INT64, ()),
                    oh.make_tensor_value_info('START', onnx.TensorProto.INT64, ()),
                ],
                [
                    oh.make_tensor_value_info('Y', onnx.TensorProto.INT64, ('NEWDIM_range',)),
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
        if node.op_type != "Range" or node.domain != "":
            return self.none()

        node_add = g.next_nodes(node.output[0])
        if len(node_add) != 1 or node_add[0].op_type != "Add" or node_add[0].domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        cst = node_add[0].input[1]
        if not g.has_shape(cst) or g.get_shape(cst) != (1,):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, node_add[0]], self.apply, insert_at=node_add[0])

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_range: NodeProto,
        node_add: NodeProto,
    ) -> List[NodeProto]:
        start, end = node_range.input[:2]

        squeezed = g.unique_name(f"{self.__class__.__name__}--{node_add.input[1]}")
        new_end = g.unique_name(f"{self.__class__.__name__}--{node_range.input[1]}")
        new_add = [
            g.make_node(
                "Squeeze",
                [node_add.input[1]],
                [squeezed],
                name=f"{self.__class__.__name__}--{node_add.name}",
                doc_string=node_add.doc_string,
            ),
            g.make_node(
                "Add",
                [end, squeezed],
                [new_end],
                name=f"{self.__class__.__name__}--{node_range.name}",
                doc_string=node_range.doc_string,
            ),
        ]

        new_range = None
        if g.is_constant(start):
            cst_start = g.get_constant_scalar(start)
            if cst_start == 0:
                new_range = g.make_node(
                    "Range",
                    [squeezed, new_end, *node_range.input[2:]],
                    [node_add.output[0]],
                    name=f"{self.__class__.__name__}--{node_range.name}",
                    doc_string=node_range.doc_string,
                )
        if new_range is None:
            new_start = g.unique_name(f"{self.__class__.__name__}--{node_range.input[0]}")
            new_add.append(
                g.make_node(
                    "Add",
                    [start, squeezed],
                    [new_start],
                    name=f"{self.__class__.__name__}--{node_range.name}",
                    doc_string=node_range.doc_string,
                )
            )
            new_range = g.make_node(
                "Range",
                [new_start, new_end, *node_range.input[2:]],
                [node_add.output[0]],
                name=f"{self.__class__.__name__}--{node_range.name}",
                doc_string=node_range.doc_string,
            )
        return [*new_add, new_range]
