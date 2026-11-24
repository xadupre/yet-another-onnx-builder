import os
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Union
from onnx_light.onnx import ModelProto, TensorProto, TypeProto, load
from .evaluator import ExtendedReferenceEvaluator

if TYPE_CHECKING:
    from ..container.export_artifact import ExportArtifact


class ReferenceNodeArg(NamedTuple):
    """Describes an input or an output like :class:`onnxruntime.NodeArg`."""

    name: str
    type: str
    shape: Optional[List[Union[int, str, None]]]


def _ort_type_name(type_proto: TypeProto) -> str:
    if type_proto.HasField("tensor_type"):
        elem_type = TensorProto.DataType(type_proto.tensor_type.elem_type).name
        return f"tensor({elem_type.lower()})"
    if type_proto.HasField("sequence_type"):
        return f"seq({_ort_type_name(type_proto.sequence_type.elem_type)})"
    if type_proto.HasField("optional_type"):
        return f"optional({_ort_type_name(type_proto.optional_type.elem_type)})"
    return "unknown"


def _ort_shape(type_proto: TypeProto) -> Optional[List[Union[int, str, None]]]:
    if not type_proto.HasField("tensor_type") or not type_proto.tensor_type.HasField("shape"):
        return None
    return [
        d.dim_value if d.HasField("dim_value") else (d.dim_param or None)
        for d in type_proto.tensor_type.shape.dim
    ]


class ReferenceInferenceSession:
    """
    Exposes :class:`yobx.reference.ExtendedReferenceEvaluator` (onnx-light runtime)
    through the subset of the :class:`onnxruntime.InferenceSession` API used by
    the unit tests. It is used when :epkg:`onnxruntime` is not installed.

    :param proto: model, serialized model, filename or export artifact
    """

    def __init__(self, proto: Union[ModelProto, bytes, str, os.PathLike, "ExportArtifact"]):
        from ..container.export_artifact import ExportArtifact

        if isinstance(proto, ExportArtifact):
            proto = proto.get_proto(include_weights=True)
        elif isinstance(proto, (str, bytes, os.PathLike)):
            proto = load(proto)
        if not isinstance(proto, ModelProto):
            raise TypeError(f"Unexpected type {type(proto)} for proto")
        self.proto = proto
        self.evaluator = ExtendedReferenceEvaluator(proto)
        initializers = {i.name for i in proto.graph.initializer}
        self._inputs = [
            ReferenceNodeArg(str(i.name), _ort_type_name(i.type), _ort_shape(i.type))
            for i in proto.graph.input
            if i.name not in initializers
        ]
        self._outputs = [
            ReferenceNodeArg(str(o.name), _ort_type_name(o.type), _ort_shape(o.type))
            for o in proto.graph.output
        ]

    def get_inputs(self) -> List[ReferenceNodeArg]:
        """Returns the model inputs."""
        return self._inputs

    def get_outputs(self) -> List[ReferenceNodeArg]:
        """Returns the model outputs."""
        return self._outputs

    def get_providers(self) -> List[str]:
        """Returns the list of providers, only the CPU is supported."""
        return ["CPUExecutionProvider"]

    def run(
        self,
        output_names: Optional[Sequence[str]],
        input_feed: Dict[str, Any],
        run_options: Any = None,
    ) -> List[Any]:
        """Runs the model and returns the requested outputs."""
        return self.evaluator.run(
            None if output_names is None else list(output_names), input_feed
        )
