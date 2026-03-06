from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable


@runtime_checkable
class TensorLike(Protocol):
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def dtype(self) -> object: ...


@runtime_checkable
class InferenceSessionLike(Protocol):
    def run(self, feeds: Dict[str, TensorLike]) -> List[TensorLike]: ...


@runtime_checkable
class GraphBuilderProtocol(Protocol):
    """Protocol describing the expected API for graph builders.

    Any class that implements this protocol can be used as a graph builder
    when converting models to ONNX format.  Both :class:`yobx.xbuilder.GraphBuilder`
    and :class:`yobx.builder.onnxscript.OnnxScriptGraphBuilder` satisfy this
    protocol.
    """

    @property
    def input_names(self) -> List[str]:
        """Returns the list of input names."""
        ...

    @property
    def output_names(self) -> List[str]:
        """Returns the list of output names."""
        ...

    def get_opset(self, domain: str, exc: bool = True) -> Optional[int]:
        """Returns the opset version for a specific domain.

        :param domain: domain name
        :param exc: raise an exception if missing
        :return: version or ``None`` when *exc* is ``False`` and the domain is unknown
        """
        ...

    def add_domain(self, domain: str, version: int = 1) -> None:
        """Registers a domain with its opset version.

        :param domain: domain name
        :param version: opset version
        """
        ...

    def unique_name(self, prefix: str) -> str:
        """Returns a unique name derived from *prefix*.

        :param prefix: name prefix
        :return: a name that has not been used yet in this graph
        """
        ...

    def has_name(self, name: str) -> bool:
        """Returns ``True`` when *name* is a known value in this graph.

        :param name: tensor name to query
        """
        ...

    def has_type(self, name: str) -> Union[bool, int]:
        """Returns ``True`` (or a truthy int) when the type of *name* is known.

        :param name: tensor name to query
        """
        ...

    def get_type(self, name: str) -> int:
        """Returns the ONNX element type of *name*.

        :param name: tensor name
        :return: ONNX element type integer
        """
        ...

    def set_type(self, name: str, itype: int) -> None:
        """Sets the ONNX element type for *name*.

        :param name: tensor name
        :param itype: ONNX element type integer
        """
        ...

    def has_shape(self, name: str) -> bool:
        """Returns ``True`` when the shape of *name* is known.

        :param name: tensor name to query
        """
        ...

    def get_shape(self, name: str) -> Tuple[Union[int, str, None], ...]:
        """Returns the shape of *name*.

        :param name: tensor name
        :return: shape as a tuple where each element is an ``int``,
            a ``str`` (symbolic dimension), or ``None`` (unknown dimension)
        """
        ...

    def set_shape(
        self,
        name: str,
        shape: Sequence[Optional[Union[int, str]]],
        allow_zero: bool = False,
    ) -> None:
        """Sets the shape for *name*.

        :param name: tensor name
        :param shape: shape to assign
        :param allow_zero: allow zero-sized dimensions
        """
        ...

    def make_tensor_input(
        self,
        name: str,
        elem_type: Optional[int] = None,
        shape: Optional[Sequence[Optional[Union[int, str]]]] = None,
        device: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """Declares a graph input and returns its name.

        :param name: tensor name
        :param elem_type: ONNX element type (e.g. ``TensorProto.FLOAT``)
        :param shape: tensor shape; use ``None`` for unknown dimensions and
            a ``str`` for symbolic / dynamic dimensions
        :param device: device identifier (optional, may be ignored by some backends)
        :return: the registered name (same as *name*)
        """
        ...

    def make_tensor_output(
        self,
        name: Union[str, List[str]],
        elem_type: Optional[int] = None,
        shape: Optional[Sequence[Optional[Union[int, str]]]] = None,
        indexed: bool = False,
        allow_untyped_output: bool = False,
    ) -> Union[str, List[str]]:
        """Declares a graph output and returns its name.

        :param name: tensor name or list of names to mark as outputs
        :param elem_type: ONNX element type hint
        :param shape: shape hint
        :param indexed: whether the name must follow an indexed naming convention
        :param allow_untyped_output: allow output with no type / shape information
        :return: the registered name or list of names
        """
        ...

    def make_initializer(
        self,
        name: str,
        value: Any,
    ) -> str:
        """Adds a constant initializer and returns its name.

        :param name: initializer name; may be empty to auto-generate a unique name
        :param value: initializer value (:class:`numpy.ndarray`,
            :class:`onnx.TensorProto`, ``int``, or ``float``)
        :return: the final registered name
        """
        ...

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, str, List[str]] = 1,
        domain: str = "",
        attributes: Optional[List[Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Creates an ONNX node and returns its output name(s).

        :param op_type: ONNX operator type (e.g. ``"Relu"``, ``"MatMul"``)
        :param inputs: input tensor name(s)
        :param outputs: number of outputs (``int``), a single output name
            (``str``), or a list of output names
        :param domain: operator domain (default ``""`` = standard ONNX)
        :param attributes: list of :class:`onnx.AttributeProto` to attach
        :param name: optional node name for debugging
        :param kwargs: operator attributes as Python primitives
        :return: output name when a single output is created, otherwise a list
        """
        ...

    def to_onnx(self) -> Any:
        """Exports the graph and returns an ONNX proto or model container.

        :return: a :class:`~onnx.ModelProto`, :class:`~onnx.GraphProto`,
            :class:`~onnx.FunctionProto`, or a model container object
        """
        ...
