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

    def has_opset(self, domain: str) -> int:
        """Returns the opset version for a domain, or 0 if the domain is not registered.

        :param domain: domain name
        :return: opset version, or ``0`` if the domain is unknown
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


@runtime_checkable
class OpsetProtocol(Protocol):
    """Protocol describing the API of an opset helper object.

    Both :class:`~yobx.xbuilder.graph_builder_opset.Opset` (used by
    :class:`~yobx.xbuilder.GraphBuilder`) and the internal opset helper
    in :class:`~yobx.builder.onnxscript.OnnxScriptGraphBuilder` satisfy this
    protocol.

    The primary usage pattern is attribute-access dispatch::

        out = g.op.Relu(x)   # equivalent to g.op.make_node("Relu", x)

    :meth:`make_node` is the core method that all opset helper implementations
    must provide.
    """

    def make_node(
        self,
        op_type: str,
        *inputs: Any,
        outputs: Optional[Union[int, List[str], str]] = None,
        domain: str = "",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, ...]]:
        """Creates an ONNX node and returns its output name(s).

        :param op_type: ONNX operator type (e.g. ``"Relu"``, ``"MatMul"``)
        :param inputs: input tensor names or constant arrays
        :param outputs: number of outputs (``int``), a single output name
            (``str``), or a list of output names; ``None`` uses a default
            inferred from *op_type* when supported
        :param domain: operator domain (default ``""`` = standard ONNX)
        :param name: optional node name for debugging
        :param kwargs: operator attributes as Python primitives
        :return: output name when a single output is created, otherwise a
            tuple of names
        """
        ...


@runtime_checkable
class GraphBuilderExtendedProtocol(GraphBuilderProtocol, Protocol):
    """Extended protocol for graph builders that support opset helpers and
    shape/type inference for common operator patterns.

    This protocol extends :class:`GraphBuilderProtocol` with additional methods
    present on :class:`yobx.xbuilder.GraphBuilder` and
    :class:`yobx.builder.onnxscript.OnnxScriptGraphBuilder` that are useful for
    advanced graph construction:

    * ``op`` — an opset helper that allows constructing nodes with
      ``g.op.Add(x, y)``-style syntax instead of calling :meth:`make_node`
      directly.
    * :meth:`set_type_shape_unary_op` — propagates type and shape from an
      input to an output for unary operators such as ``Abs``, ``Relu``, etc.
    """

    @property
    def op(self) -> "OpsetProtocol":
        """Returns the opset helper for this graph builder.

        The opset helper allows constructing ONNX nodes using attribute-access
        syntax.  For example::

            out = g.op.Relu(x)

        :return: an :class:`OpsetProtocol`-compatible object
        """
        ...

    def set_type_shape_unary_op(
        self,
        name: str,
        input_name: str,
        itype: Optional[int] = None,
    ) -> bool:
        """Propagates type and shape from *input_name* to *name* for a unary op.

        This is a convenience helper used when emitting elementwise unary
        operators (``Abs``, ``Exp``, ``Relu``, etc.) where the output has the
        same type and shape as the input.

        :param name: output tensor name whose type/shape should be set
        :param input_name: input tensor name to copy type/shape from
        :param itype: override element type; if ``None`` the input's element
            type is used
        :return: ``True`` when shape information was available and set,
            ``None``/falsy when it could not be determined
        """
        ...
