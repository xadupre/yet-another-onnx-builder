from typing import Any, Dict, List
from yobx._onnx_shim import onnx
from ..translate.translator import Translator
from ..translate.builder_emitter import BuilderEmitter


class CustomBuilderEmitter(BuilderEmitter):
    """Custom :class:`yobx.translate.builder_emitter.BuilderEmitter`."""

    def __init__(self, make_model_function: str = "make_my_model"):
        super().__init__(make_model_function=make_model_function)

    def _emit_node_type(self, op_type, op_domain):
        if op_type in {"Squeeze", "Unsqueeze"} or op_type.startswith("Reduce"):
            return f"{op_type}AnyOpset"
        return op_type

    def _clean_result_name(self, name):
        return name.replace("#", "__").replace("-", "_")

    def _emit_end_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        rows = super()._emit_end_function(**kwargs)
        return [
            *rows[:-1],
            "    opts = FunctionOptions(",
            f"        name={self.f_name!r},",
            f"        domain={self.f_domain!r},",
            "        move_initializer_to_constant=True,",
            "    )",
            "    g.make_local_function(gr, opts, optimize=False)",
        ]


def to_graph_builder_code(proto: onnx.ModelProto, function_name: str = "build_model") -> str:
    """
    Produces a code building a model with
    :class:`yobx.xbuilder.GraphBuilder`.

    :param proto: model to convert into a code
    :param function_name: function name
    :return: str

    .. runpython::
        :showcode:

        import numpy as np
        from yobx._onnx_shim import onnx
        import onnx_light.onnx.helper as oh
        import onnx_light.onnx.numpy_helper as onh
        from yobx.translate.reverse_graph_builder import (
            to_graph_builder_code,
        )

        TFLOAT = onnx.TensorProto.FLOAT
        TINT64 = onnx.TensorProto.INT64

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["cst"],
                        value=onh.from_array(np.array([0], dtype=np.float32)),
                    ),
                    oh.make_node(
                        "ScatterND",
                        ["cst", "indices", "updates"],
                        ["Z"],
                        reduction="add",
                    ),
                ],
                "create_graph",
                [
                    oh.make_tensor_value_info("shape", TINT64, [None]),
                    oh.make_tensor_value_info("indices", TINT64, [None, None]),
                    oh.make_tensor_value_info("updates", TFLOAT, [None, None, None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        print(to_graph_builder_code(model))
    """
    tr = Translator(proto, emitter=CustomBuilderEmitter(make_model_function=function_name))
    code = tr.export(as_str=True)
    return "\n".join(
        [
            "import numpy as np",
            "from yobx._onnx_shim import onnx",
            "import onnx_light.onnx.numpy_helper as onh",
            "from yobx.xbuilder import GraphBuilder, FunctionOptions",
            "",
            "",
            code.replace("array(nan", "array(np.nan"),
        ]
    )
