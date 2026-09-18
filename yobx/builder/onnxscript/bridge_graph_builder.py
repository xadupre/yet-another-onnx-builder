"""Retires the bridge whose third-party runtime requires reference ONNX."""


class OnnxScriptGraphBuilder:
    """Rejects the removed reference-ONNX-dependent builder."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The onnxscript builder requires the removed reference ONNX dependency. "
            "Use yobx.xbuilder.GraphBuilder, backed by onnx-light."
        )
