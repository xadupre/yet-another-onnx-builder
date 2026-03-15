from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)


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

    def get_opset(self, domain: str, exc: bool = True) -> int:
        """Returns the opset version for a specific domain.

        :param domain: domain name
        :param exc: raise an exception if missing
        :return: version or 0 when *exc* is ``False`` and the domain is unknown
        """
        ...

    def set_opset(self, domain: str, version: int = 1) -> None:
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

    def get_shape(self, name: str) -> Tuple[Union[int, str], ...]:
        """Returns the shape of *name*.

        :param name: tensor name
        :return: shape as a tuple where each element is an ``int``,
            a ``str`` (symbolic dimension)
        """
        ...

    def set_shape(
        self, name: str, shape: Tuple[Union[int, str], ...], allow_zero: bool = False
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
        shape: Optional[Tuple[Union[int, str], ...]] = None,
        device: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """Declares a graph input and returns its name.

        :param name: tensor name
        :param elem_type: ONNX element type (e.g. ``TensorProto.FLOAT``)
        :param shape: tensor shape; use a ``str`` for symbolic or unknown
            dimensions (e.g. ``"batch"`` or ``"?"``), and an ``int`` for
            fixed-size dimensions
        :param device: device identifier (optional, may be ignored by some backends)
        :return: the registered name (same as *name*)
        """
        ...

    def make_tensor_output(
        self,
        name: Union[str, List[str]],
        elem_type: Optional[int] = None,
        shape: Optional[Tuple[Union[int, str], ...]] = None,
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

    def make_initializer(self, name: str, value: Any) -> str:
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
    ) -> Union[str, Tuple[str]]:
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

        out = g.op.Relu(x)   # resolves via __getattr__("Relu"), then calls result

    ``__getattr__`` must return a callable that, when invoked, creates the
    corresponding ONNX node and returns its output name(s).
    """

    def __getattr__(self, op_type: str) -> Callable[..., Union[str, Tuple[str, ...]]]:
        """Returns a callable that creates an ONNX node of type *op_type*.

        :param op_type: ONNX operator type name (e.g. ``"Relu"``, ``"MatMul"``)
        :return: a callable that accepts tensor inputs and keyword attributes
            and returns the output name (``str``) or a tuple of output names
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

    * ``main_opset`` — the opset version for the main (``""``/ONNX) domain.
    * ``op`` — an opset helper that allows constructing nodes with
      ``g.op.Add(x, y)``-style syntax via :meth:`~OpsetProtocol.__getattr__`.
    * :meth:`set_type_shape_unary_op` — propagates type and shape from an
      input to an output for unary operators such as ``Abs``, ``Relu``, etc.
    """

    @property
    def main_opset(self) -> int:
        """Returns the opset version for the main (``""``/ONNX) domain.

        :return: integer opset version for domain ``""``
        """
        ...

    def unique_name(self, prefix: str) -> str:
        """Returns a unique name derived from *prefix*.

        :param prefix: name prefix
        :return: a name that has not been used yet in this graph
        """
        ...

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
        self, name: str, input_name: str, itype: Optional[int] = None
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

    def get_debug_msg(self) -> str:
        """Returns any information useful to understand where an error
        could come from. This message is expected to be part of any
        exception raised while converting a model.

        :return: information in a string
        """
        ...


@runtime_checkable
class GraphBuilderPatternOptimizationProtocol(Protocol):
    """Protocol describing the expected API of the graph pattern optimizer.

    Any class that implements this protocol can be passed as the *g* argument
    to :meth:`~yobx.xoptim.PatternOptimization.match` and related methods.
    The concrete implementation is
    :class:`yobx.xoptim.GraphBuilderPatternOptimization`.

    The protocol covers all attributes and methods that pattern authors
    typically call on the optimizer object inside their ``match()``
    implementations.
    """

    # ------------------------------------------------------------------
    # Instance attributes surfaced as Protocol members
    # ------------------------------------------------------------------

    @property
    def verbose(self) -> int:
        """Current verbosity level."""
        ...

    @property
    def processor(self) -> str:
        """Target processor(s), e.g. ``"CPU"`` or ``"CPU,CUDA"``."""
        ...

    @property
    def builder(self) -> Any:
        """The underlying :class:`~yobx.xbuilder.GraphBuilder` instance."""
        ...

    # ------------------------------------------------------------------
    # Graph-level properties
    # ------------------------------------------------------------------

    @property
    def main_opset(self) -> int:
        """Opset version for the main (``""``/ONNX) domain.

        :return: integer opset version
        """
        ...

    @property
    def opsets(self) -> Dict[str, int]:
        """Mapping from domain name to opset version.

        :return: dict of domain → opset version
        """
        ...

    @property
    def nodes(self) -> List[Any]:
        """Ordered list of :class:`~onnx.NodeProto` objects in the graph.

        :return: list of nodes
        """
        ...

    @property
    def input_names(self) -> List[str]:
        """Names of the graph inputs.

        :return: list of input names
        """
        ...

    @property
    def output_names(self) -> List[str]:
        """Names of the graph outputs.

        :return: list of output names
        """
        ...

    @property
    def inputs(self) -> List[Any]:
        """Graph input value infos.

        :return: list of :class:`~onnx.ValueInfoProto` objects
        """
        ...

    @property
    def outputs(self) -> List[Any]:
        """Graph output value infos.

        :return: list of :class:`~onnx.ValueInfoProto` objects
        """
        ...

    # ------------------------------------------------------------------
    # Node navigation
    # ------------------------------------------------------------------

    def iter_nodes(self) -> Iterator[Any]:
        """Iterates over all nodes in the graph.

        :return: iterator of :class:`~onnx.NodeProto`
        """
        ...

    def node_before(self, name: str) -> Optional[Any]:
        """Returns the node that produces output *name*, or ``None``.

        :param name: result name
        :return: :class:`~onnx.NodeProto` or ``None`` if *name* is an input
            or initializer
        """
        ...

    def next_node(self, name: str) -> Any:
        """Returns the unique consumer of *name*.  Raises if there is not
        exactly one consumer.

        :param name: result name
        :return: :class:`~onnx.NodeProto`
        """
        ...

    def next_nodes(self, name: str) -> List[Any]:
        """Returns all nodes that consume *name*.

        :param name: result name
        :return: list of :class:`~onnx.NodeProto`
        """
        ...

    def get_position(self, node: Any) -> int:
        """Returns the position (index) of *node* in the graph node list.

        :param node: a :class:`~onnx.NodeProto`
        :return: zero-based index
        """
        ...

    # ------------------------------------------------------------------
    # Liveness / usage queries
    # ------------------------------------------------------------------

    def is_used(self, name: str) -> bool:
        """Returns ``True`` when *name* is consumed by any node or is a
        graph output.

        :param name: result name
        """
        ...

    def is_used_more_than_once(self, name: str) -> bool:
        """Returns ``True`` when *name* has more than one consumer, is a
        graph output, or is referenced by a sub-graph.

        :param name: result name
        """
        ...

    def is_used_only_by(self, name: str, *nodes: Any) -> bool:
        """Returns ``True`` when *name* is consumed exclusively by the
        given nodes.

        :param name: result name
        :param nodes: the only permitted consumers
        """
        ...

    def is_output(self, name: str) -> bool:
        """Returns ``True`` when *name* is a graph output.

        :param name: result name
        """
        ...

    def is_used_by_subgraph(self, name: str) -> bool:
        """Returns ``True`` when *name* is used inside a sub-graph.

        :param name: result name
        """
        ...

    # ------------------------------------------------------------------
    # Constant queries
    # ------------------------------------------------------------------

    def is_constant(self, name: str) -> bool:
        """Returns ``True`` when *name* is a known constant (initializer
        or the output of a constant-foldable node chain).

        :param name: result name
        """
        ...

    def is_constant_scalar(
        self, name: str, value: Optional[Any] = None, broadcast: bool = False
    ) -> bool:
        """Returns ``True`` when *name* is a scalar constant.

        :param name: result name
        :param value: if not ``None``, also check that the scalar equals
            this value
        :param broadcast: treat shapes ``(1,)``, ``(1,1)``, … as scalar
        """
        ...

    def get_constant_shape(
        self, name: str, exc: bool = True
    ) -> Optional[Tuple[int, ...]]:
        """Returns the shape of constant *name*.

        :param name: result name
        :param exc: raise an exception if the shape cannot be determined
        :return: shape tuple, or ``None`` when *exc* is ``False``
        """
        ...

    def get_computed_constant(
        self, name: str, statistics: Optional[List[str]] = None
    ) -> Any:
        """Returns the evaluated value of constant *name*.

        :param name: result name
        :param statistics: optional list of summary statistics to compute
            (``"min"``, ``"max"``); when given, a list of values is returned
        :return: :class:`numpy.ndarray` or a list of statistics
        """
        ...

    def get_constant_scalar(
        self, name: str, broadcast: bool = False
    ) -> Union[int, float]:
        """Returns the scalar value of constant *name*.

        :param name: result name
        :param broadcast: accept shapes such as ``(1,)`` or ``(1,1)``
        :return: ``int``, ``float``, or ``complex``
        """
        ...

    def get_constant_or_attribute(
        self,
        node: Any,
        attribute: str,
        input_index: int,
        cvt: Optional[Callable] = None,
    ) -> Any:
        """Returns the value of an operator attribute or input depending on
        the opset version.  Some attributes became inputs in newer opsets.

        :param node: :class:`~onnx.NodeProto`
        :param attribute: attribute name (used in older opsets)
        :param input_index: input index (used in newer opsets)
        :param cvt: optional conversion callable applied to the result
        :return: attribute or constant value
        """
        ...

    # ------------------------------------------------------------------
    # Type / shape queries
    # ------------------------------------------------------------------

    def has_type(self, name: str) -> bool:
        """Returns ``True`` when the element type of *name* is known.

        :param name: result name
        """
        ...

    def get_type(self, name: str) -> int:
        """Returns the ONNX element type integer for *name*.

        :param name: result name
        :return: element type (e.g. ``TensorProto.FLOAT``)
        """
        ...

    def has_rank(self, name: str) -> bool:
        """Returns ``True`` when the rank of *name* is known.

        :param name: result name
        """
        ...

    def get_rank(self, name: str) -> int:
        """Returns the rank of *name*.

        :param name: result name
        :return: number of dimensions
        """
        ...

    def has_shape(self, name: str) -> bool:
        """Returns ``True`` when the full shape of *name* is known.

        :param name: result name
        """
        ...

    def get_shape(self, name: str) -> Tuple[Union[int, str], ...]:
        """Returns the shape of *name*.

        :param name: result name
        :return: tuple where each element is an ``int`` or a symbolic
            dimension string
        """
        ...

    def same_shape(self, a: str, b: str) -> bool:
        """Returns ``True`` when *a* and *b* have the same shape,
        taking registered constraints into account.

        :param a: first result name
        :param b: second result name
        """
        ...

    def get_shape_renamed(self, name: str) -> Tuple[Union[int, str], ...]:
        """Returns the shape of *name* using user-visible dimension names.

        :param name: result name
        :return: shape tuple with user dimension names
        """
        ...

    def try_infer_type(self, name: str, exc: bool = False) -> int:
        """Tries to infer the element type of *name*, propagating through
        the graph if necessary.

        :param name: result name
        :param exc: raise an exception when the type cannot be determined
        :return: element type integer, or ``0`` when unknown
        """
        ...

    def try_infer_shape(
        self, name: str, exc: bool = False
    ) -> Optional[Tuple[Union[int, str], ...]]:
        """Tries to infer the shape of *name*.

        :param name: result name
        :param exc: raise an exception when the shape cannot be determined
        :return: shape tuple or ``None``
        """
        ...

    # ------------------------------------------------------------------
    # Attribute helpers
    # ------------------------------------------------------------------

    def get_attribute(
        self, node: Any, att_name: str, exc: bool = True
    ) -> Optional[Any]:
        """Returns the :class:`~onnx.AttributeProto` named *att_name* on
        *node*.

        :param node: :class:`~onnx.NodeProto`
        :param att_name: attribute name
        :param exc: raise an exception if the attribute is missing
        :return: :class:`~onnx.AttributeProto` or ``None``
        """
        ...

    def get_attribute_with_default(
        self, node: Any, name: str, default_value: Any
    ) -> Any:
        """Returns the value of attribute *name* on *node*, or
        *default_value* if the attribute is absent.

        :param node: :class:`~onnx.NodeProto`
        :param name: attribute name
        :param default_value: fallback value
        :return: attribute value or *default_value*
        """
        ...

    def get_attributes_with_default(
        self, node: Any, **default_values: Any
    ) -> Dict[str, Any]:
        """Returns a dict of attribute values for *node*, substituting
        *default_values* for any missing attributes.

        :param node: :class:`~onnx.NodeProto`
        :param default_values: keyword arguments mapping attribute names to
            their default values
        :return: dict of attribute name → value
        """
        ...

    def get_axis(self, node: Any, default_axis: Optional[int] = None) -> int:
        """Returns the ``axis`` attribute of *node*.

        :param node: :class:`~onnx.NodeProto`
        :param default_axis: default value when the attribute is absent
        :return: axis integer
        """
        ...

    # ------------------------------------------------------------------
    # Processor / constraint helpers
    # ------------------------------------------------------------------

    def has_processor(self, processor: str) -> bool:
        """Returns ``True`` when *processor* is in the active processor list.

        :param processor: processor name, e.g. ``"CUDA"``
        """
        ...

    def get_registered_constraints(self) -> Dict[str, Set[Union[str, int]]]:
        """Returns the shape constraints registered on the builder.

        :return: mapping from constraint name to set of allowed values
        """
        ...

    def has_exact_same_constant_in_context(
        self, name: str
    ) -> Optional[bool]:
        """Checks whether an identical constant already exists in the graph.

        :param name: constant name to look up
        :return: ``True``/``False`` when known, ``None`` otherwise
        """
        ...

    def do_not_turn_constant_initializers_maybe_because_of_showing(
        self, name: str
    ) -> bool:
        """Returns ``True`` when the initializer for *name* must not be
        folded into a ``Constant`` node (e.g. because it is displayed).

        :param name: initializer name
        """
        ...

    # ------------------------------------------------------------------
    # Node / initializer creation
    # ------------------------------------------------------------------

    def make_initializer(
        self,
        name: str,
        value: Any,
        external: bool = False,
        msg: str = "",
        source: Optional[str] = None,
        give_unique: bool = True,
    ) -> str:
        """Adds a constant initializer and returns its (possibly
        auto-generated) name.

        :param name: desired name (may be empty for auto-naming)
        :param value: :class:`numpy.ndarray` or
            :class:`~onnx.TensorProto` value
        :param external: store as external data
        :param msg: optional debug message
        :param source: optional source tag for debugging
        :param give_unique: generate a unique name when *name* is already
            taken
        :return: the registered name
        """
        ...

    def unique_name(self, prefix: str) -> str:
        """Returns a name derived from *prefix* that has not been used yet.

        :param prefix: name prefix
        :return: unique name
        """
        ...

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Creates an ONNX node (without adding it to the graph) and
        returns the :class:`~onnx.NodeProto`.

        :param op_type: operator type
        :param inputs: input name(s)
        :param outputs: number of outputs (``int``), single output name
            (``str``), or list of output names
        :param domain: operator domain
        :param attributes: list of :class:`~onnx.AttributeProto` objects
        :param name: node name
        :param kwargs: operator attributes as Python primitives
        :return: :class:`~onnx.NodeProto`
        """
        ...

    def make_node_check_opset(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Like :meth:`make_node` but adapts certain operators for the
        active opset (e.g. ``Squeeze``/``Unsqueeze`` axis handling changed
        between opset 11 and 13).

        :param op_type: operator type
        :param inputs: input name(s)
        :param outputs: number of outputs (``int``), single output name
            (``str``), or list of output names
        :param domain: operator domain (must be ``""`` for the main domain)
        :param attributes: list of :class:`~onnx.AttributeProto` objects
        :param name: node name
        :param kwargs: operator attributes as Python primitives
        :return: :class:`~onnx.NodeProto`
        """
        ...

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def pretty_text(
        self, add_fx_graph: bool = False, recursive: bool = True
    ) -> str:
        """Returns a human-readable text rendering of the graph.

        :param add_fx_graph: include the FX graph representation when
            available
        :param recursive: recurse into sub-graphs
        :return: multi-line string
        """
        ...
