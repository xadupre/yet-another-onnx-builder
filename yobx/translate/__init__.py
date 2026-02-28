import textwrap
from onnx import ModelProto
from .translate import Translater
from .inner_emitter import InnerEmitter, InnerEmitterShortInitializer
from .builder_emitter import BuilderEmitter
from .light_emitter import LightEmitter


def translate_header(api: str = "onnx"):
    """Returns the necessary imports header for each api."""
    if api == "onnx":
        return textwrap.dedent(
            """
            import numpy as np
            import ml_dtypes
            import onnx
            import onnx.helper as oh
            import onnx.numpy_helper as onh
            from yobx.translate.make_helper import make_node_extended
            """
        )
    if api == "onnx-short":
        return textwrap.dedent(
            """
            import numpy as np
            import ml_dtypes
            import onnx
            import onnx.helper as oh
            import onnx.numpy_helper as onh
            from yobx.translate.make_helper import make_node_extended
            """
        )
    if api == "light":
        return textwrap.dedent(
            """
            import numpy as np
            import ml_dtypes
            import onnx
            from onnx_array_api.light_api import start
            from yobx.translate import translate
            """
        )
    if api == "builder":
        return textwrap.dedent(
            """
            import numpy as np
            import ml_dtypes
            import onnx
            from onnx_array_api.graph_api import GraphBuilder
            """
        )
    raise ValueError(f"Unexpected value {api!r} for api.")


def translate(proto: ModelProto, single_line: bool = False, api: str = "onnx") -> str:
    """
    Translates an ONNX proto into Python code that recreates the ONNX graph.

    :param proto: model to translate
    :param single_line: as a single line or not
    :param api: API to export into,
        default is ``"onnx"`` which generates code using ``onnx.helper``
        (handled by :class:`~yobx.translate.inner_emitter.InnerEmitter`),
        ``"onnx-short"`` replaces large initializers with random values
        (handled by
        :class:`~yobx.translate.inner_emitter.InnerEmitterShortInitializer`),
        ``"light"`` generates code for the ``onnx-array-api`` light API
        (requires ``onnx-array-api`` to be installed),
        ``"builder"`` generates code for the ``onnx-array-api`` GraphBuilder
        (requires ``onnx-array-api`` to be installed)
    :return: code as a string
    """
    if api == "onnx":
        tr = Translater(proto, emitter=InnerEmitter())
        return tr.export(as_str=True)
    if api == "onnx-short":
        tr = Translater(proto, emitter=InnerEmitterShortInitializer())
        return tr.export(as_str=True)
    if api == "light":
        tr = Translater(proto, emitter=LightEmitter())
        return tr.export(single_line=single_line, as_str=True)
    if api == "builder":
        tr = Translater(proto, emitter=BuilderEmitter())
        return tr.export(as_str=True)
    raise ValueError(f"Unexpected value {api!r} for api.")
