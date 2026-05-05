from typing import Dict
import onnx
from .onnx_helper import onnx_dtype_name


def to_mermaid(model: onnx.ModelProto) -> str:
    """
    Converts a model into a `Mermaid <https://mermaid.js.org/>`_ ``flowchart TD`` string.

    The function:

    * uses :class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
      to annotate every edge with its inferred dtype and shape (when available),
    * inlines small scalar constants and 1-D initializers directly onto the node
      label so the graph stays compact,
    * handles ``Scan`` / ``Loop`` / ``If`` sub-graphs by drawing dotted edges for
      outer-scope values consumed by the sub-graph.

    :param model: ONNX model to convert
    :returns: Mermaid flowchart string

    .. runpython::
        :showcode:

        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from yobx.helpers.mermaid_helper import to_mermaid

        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["added"]),
                    oh.make_node("MatMul", ["added", "W"], ["mm"]),
                    oh.make_node("Relu", ["mm"], ["Z"]),
                ],
                "add_matmul_relu",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 4]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", 4]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "seq", 2])],
                [onh.from_array(np.zeros((4, 2), dtype=np.float32), name="W")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        mermaid_src = to_mermaid(model)
        print(mermaid_src)

    .. runmermaid::
        :script:

        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from yobx.helpers.mermaid_helper import to_mermaid

        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["added"]),
                    oh.make_node("MatMul", ["added", "W"], ["mm"]),
                    oh.make_node("Relu", ["mm"], ["Z"]),
                ],
                "add_matmul_relu",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 4]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", 4]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "seq", 2])],
                [onh.from_array(np.zeros((4, 2), dtype=np.float32), name="W")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        print(to_mermaid(model))
    """
    from ..xshape import BasicShapeBuilder

    builder = BasicShapeBuilder()
    builder.run_model(model)

    edge_labels: Dict[str, str] = {}
    for node in model.graph.node:
        for name in node.output:
            if name and builder.has_type(name) and builder.has_shape(name):
                itype = builder.get_type(name)
                if itype == onnx.TensorProto.UNDEFINED:
                    continue
                shape = builder.get_shape(name)
                res = [
                    str(a)
                    for a in [
                        ("?" if isinstance(s, str) and s.startswith("unk") else s) for s in shape
                    ]
                ]
                sshape = ",".join(res)
                edge_labels[name] = f"{onnx_dtype_name(itype)}({sshape})"

    from ..translate.mermaid_emitter import MermaidEmitter
    from ..translate.translator import Translator

    emitter = MermaidEmitter(edge_labels=edge_labels)
    tr = Translator(model, emitter=emitter)  # type: ignore
    return tr.export(as_str=True)  # type: ignore
