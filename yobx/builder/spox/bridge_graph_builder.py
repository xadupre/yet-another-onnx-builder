"""Retires the bridge whose third-party runtime requires reference ONNX."""


class SpoxGraphBuilder:
    """Rejects the removed reference-ONNX-dependent builder."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The Spox builder requires the removed reference ONNX dependency. "
            "Use yobx.xbuilder.GraphBuilder, backed by onnx-light."
        )
