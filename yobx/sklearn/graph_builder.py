"""Provides the onnx-light graph builder used by scikit-learn converters."""

from onnx_light import onnx

from ..builder.onnxlight import OnnxLightGraphBuilder


class SklearnOnnxLightGraphBuilder(OnnxLightGraphBuilder):
    """Adds scikit-learn converter naming compatibility to the native builder."""

    def __init__(self, target_opset_or_existing_proto=18, *args, **kwargs):
        """Initializes the builder and converts foreign ONNX model protobufs."""
        descriptor = getattr(target_opset_or_existing_proto, "DESCRIPTOR", None)
        if (
            not isinstance(target_opset_or_existing_proto, onnx.ModelProto)
            and getattr(descriptor, "full_name", None) == "onnx.ModelProto"
        ):
            model = onnx.ModelProto()
            model.ParseFromString(target_opset_or_existing_proto.SerializeToString())
            target_opset_or_existing_proto = model
        super().__init__(target_opset_or_existing_proto, *args, **kwargs)

    def unique_node_name(self, prefix: str) -> str:
        """Returns a collision-free operator name."""
        return self.unique_name(prefix)

    def unique_function_name(self, prefix: str) -> str:
        """Returns a collision-free local-function name."""
        return self.unique_name(prefix)
