from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
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
class GraphBuilderTorchProtocol(GraphBuilderExtendedProtocol, Protocol):
    """Protocol for graph builders that support the full torch-exporter API.

    This protocol extends :class:`GraphBuilderExtendedProtocol` with the
    additional methods and attributes used by
    :class:`~yobx.torch.interpreter.DynamoInterpreter` (the *torch exporter*)
    when translating a ``torch.fx`` graph into ONNX.

    The extra surface covers:

    * **Rank helpers** — :meth:`has_rank`, :meth:`get_rank`, :meth:`set_rank`.
    * **Device helpers** — :meth:`has_device`, :meth:`get_device`,
      :meth:`set_device`.
    * **Extended type / shape** — :meth:`get_type_known`,
      :meth:`set_shapes_types`.
    * **Tensor-sequence support** — :meth:`is_sequence`, :meth:`get_sequence`,
      :meth:`set_sequence`, :meth:`make_tensor_sequence_input`.
    * **Dynamic-shape helpers** — :meth:`is_dynamic_shape`,
      :meth:`get_input_dynamic_shape`, :meth:`get_is_dimension`,
      :meth:`verify_dynamic_shape`, :meth:`register_dynamic_objects_from_shape`,
      :meth:`make_dynamic_object`, :meth:`add_dynamic_object`,
      :meth:`make_new_dynamic_shape`.
    * **Sub-builder / local-function support** — :meth:`make_nodes`,
      :meth:`make_local_function`, :meth:`make_subset_builder`.
    * **Miscellaneous** — :meth:`add_stat`, :meth:`pretty_text`,
      :meth:`register_users`, :meth:`extract_input_names_from_args`.
    * **State attributes** — ``anyop``, ``as_function``, ``local_domain``,
      ``verbose``, ``torch``, ``optimization_options``, ``dynamic_shapes``,
      ``dynamic_objects``, ``dynamic_dimensions_source``, ``nodes``,
      ``outputs``, ``opsets``, ``initializers_dict``, ``raise_list``,
      ``was_inputs_renamed``, ``last_added_node``.
    """

    # ------------------------------------------------------------------
    # Attributes / properties
    # ------------------------------------------------------------------

    @property
    def anyop(self) -> "OpsetProtocol":
        """An opset helper that allows operators from any domain.

        Identical to :attr:`~GraphBuilderExtendedProtocol.op` but configured
        to accept unknown domains.  Used by the torch exporter to emit
        custom-domain operators such as ``ai.onnx.complex``.

        :return: an :class:`OpsetProtocol`-compatible object
        """
        ...

    as_function: bool
    """``True`` when the graph is being exported as a local function."""

    local_domain: str
    """Domain name used for local functions created during the export."""

    verbose: int
    """Verbosity level (``0`` = silent)."""

    torch: Any
    """Reference to the :mod:`torch` module, or ``None`` when PyTorch is
    not installed."""

    optimization_options: Any
    """Optimization options forwarded to the builder's optimisation pass."""

    dynamic_shapes: Any
    """Dynamic-shapes specification provided to the exporter, or ``None``."""

    dynamic_objects: Dict[str, Any]
    """Map of dynamic dimension names (strings) to their current values
    (e.g. :class:`torch.SymInt` instances or :class:`int` literals)."""

    dynamic_dimensions_source: Dict[str, Any]
    """Map of dynamic dimension names to the graph inputs / axes they
    originate from."""

    nodes: List[Any]
    """Ordered list of ONNX :class:`~onnx.NodeProto` objects accumulated
    so far."""

    outputs: List[Any]
    """List of declared graph outputs."""

    opsets: Dict[str, int]
    """Map of ONNX domain names to their opset versions."""

    initializers_dict: Dict[str, Any]
    """Map of initializer names to their :class:`~onnx.TensorProto` values."""

    raise_list: Optional[Set[str]]
    """When set, the builder raises an exception if a result in this set is
    produced — useful for debugging."""

    was_inputs_renamed: bool
    """``True`` when the builder renamed some graph inputs during
    construction."""

    @property
    def last_added_node(self) -> Optional[Any]:
        """The most recently appended ONNX node, or ``None`` when the graph
        is still empty.

        :return: an :class:`~onnx.NodeProto` or ``None``
        """
        ...

    # ------------------------------------------------------------------
    # Rank helpers
    # ------------------------------------------------------------------

    def has_rank(self, name: str) -> bool:
        """Returns ``True`` when the rank of tensor *name* is known.

        :param name: tensor name
        """
        ...

    def get_rank(self, name: str) -> int:
        """Returns the rank (number of dimensions) of tensor *name*.

        :param name: tensor name
        :return: non-negative integer rank
        """
        ...

    def set_rank(self, name: str, value: int) -> None:
        """Records the rank for tensor *name*.

        :param name: tensor name
        :param value: rank (number of dimensions)
        """
        ...

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    def has_device(self, name: str) -> bool:
        """Returns ``True`` when the device of tensor *name* is known.

        :param name: tensor name
        """
        ...

    def get_device(self, name: str) -> int:
        """Returns the device index for tensor *name*.

        The convention is ``-1`` for CPU and a non-negative index for a
        CUDA device.

        :param name: tensor name
        :return: device index
        """
        ...

    def set_device(
        self,
        name: str,
        device: Any,
        exc: bool = True,
        keep_this_device: bool = False,
    ) -> None:
        """Records the device for tensor *name*.

        :param name: tensor name
        :param device: device identifier — an :class:`int` index, a
            :class:`torch.device`, or a constant name
        :param exc: raise an exception on inconsistency
        :param keep_this_device: overwrite an already-known device
        """
        ...

    # ------------------------------------------------------------------
    # Extended type / shape knowledge
    # ------------------------------------------------------------------

    def get_type_known(self, name: str, exc: bool = False) -> Optional[int]:
        """Returns the ONNX element type inferred by torch for *name*, or
        ``None`` when unavailable.

        :param name: tensor name
        :param exc: raise an exception when the value is malformed
        :return: ONNX element type integer or ``None``
        """
        ...

    def set_shapes_types(self, name: Any, where: str, value: Any) -> None:
        """Records a torch-side ``(where, value)`` annotation for *name*.

        The annotation is later consulted by :meth:`get_type_known` to
        resolve type mismatches between ONNX and torch.

        :param name: tensor name (or a :class:`torch.fx.Node`)
        :param where: source label (e.g. ``"run_node"``)
        :param value: annotation value (a tuple describing type / shape)
        """
        ...

    # ------------------------------------------------------------------
    # Tensor-sequence support
    # ------------------------------------------------------------------

    def is_sequence(self, name: str) -> bool:
        """Returns ``True`` when *name* has been registered as a tensor
        sequence.

        :param name: tensor name
        """
        ...

    def get_sequence(self, name: str) -> Dict[str, Any]:
        """Returns sequence metadata for *name*.

        :param name: tensor name
        :return: dictionary with keys such as ``"dtype"``, ``"shapes"``,
            ``"ranks"``
        """
        ...

    def set_sequence(
        self,
        name: str,
        dtype: Any,
        shapes: Optional[Any] = None,
        ranks: Optional[Any] = None,
        unknown: bool = False,
    ) -> None:
        """Marks *name* as a tensor sequence.

        :param name: tensor name
        :param dtype: element type (ONNX integer or tuple of integers)
        :param shapes: optional tuple of per-element shapes
        :param ranks: optional tuple of per-element ranks
        :param unknown: set ``True`` when sequence contents are unknown
        """
        ...

    def make_tensor_sequence_input(
        self,
        name: str,
        elem_type: Any,
        shape: Any,
        marker: str = "",
    ) -> str:
        """Registers a tensor-sequence graph input and returns its name.

        :param name: desired input name
        :param elem_type: ONNX element type for the sequence elements
        :param shape: shape of each element
        :param marker: optional label for debugging / provenance
        :return: the registered input name
        """
        ...

    # ------------------------------------------------------------------
    # Dynamic-shape helpers
    # ------------------------------------------------------------------

    def is_dynamic_shape(
        self,
        shape: Any,
        verify: bool = True,
        allow_none: bool = False,
        allow_new_dynamic_dimension: bool = False,
    ) -> bool:
        """Returns ``True`` when *shape* contains at least one dynamic
        (non-integer) dimension.

        :param shape: shape tuple to check
        :param verify: verify that symbolic names are registered
        :param allow_none: treat ``None`` entries as dynamic
        :param allow_new_dynamic_dimension: allow unregistered symbolic dims
        """
        ...

    def get_input_dynamic_shape(
        self,
        name: Optional[str],
        input_index: int,
        example_shape: Any,
        dynamic_shapes: Optional[Any] = None,
        example_value: Optional[Any] = None,
    ) -> Any:
        """Returns the dynamic shape specification for a graph input.

        :param name: input name
        :param input_index: positional index of the input
        :param example_shape: concrete shape of an example input
        :param dynamic_shapes: override the builder's ``dynamic_shapes``
        :param example_value: one example tensor value
        :return: shape tuple (mixing ``int`` for static dimensions and
            ``str`` for dynamic ones)
        """
        ...

    def get_is_dimension(
        self,
        name: str,
        elem_type: Optional[int] = None,
        shape: Optional[Any] = None,
        n_outputs: Optional[int] = None,
        exc: bool = True,
    ) -> bool:
        """Returns ``True`` when *name* represents a scalar dynamic-dimension
        value (e.g. the result of ``aten.sym_size``).

        :param name: tensor name
        :param elem_type: expected ONNX element type hint
        :param shape: expected shape hint
        :param n_outputs: number of outputs produced by the enclosing node
        :param exc: raise when ambiguous
        """
        ...

    def verify_dynamic_shape(
        self,
        shape: Any,
        name: Optional[str] = None,
        add: bool = True,
    ) -> Optional[Any]:
        """Normalises *shape*, replacing symbolic dimensions with their
        registered string names.

        :param shape: raw shape (may contain :class:`torch.SymInt`)
        :param name: tensor name for context messages
        :param add: register newly encountered symbolic dims automatically
        :return: normalised shape tuple or ``None`` if *shape* is ``None``
        """
        ...

    def register_dynamic_objects_from_shape(self, shape: Any) -> None:
        """Registers all dynamic-dimension objects found in *shape*.

        :param shape: shape tuple potentially containing symbolic dimensions
        """
        ...

    def make_dynamic_object(
        self,
        name: str,
        value: Any,
        shape_as_input: bool = False,
        input_name: Optional[str] = None,
        axis: Optional[int] = None,
    ) -> Optional[str]:
        """Creates and registers a dynamic dimension object.

        :param name: name for the dynamic dimension
        :param value: :class:`torch.SymInt` or similar symbolic value
        :param shape_as_input: add the dimension as a scalar graph input
        :param input_name: the tensor input this dimension originates from
        :param axis: the axis of *input_name* this dimension corresponds to
        :return: the registered name, or ``None``
        """
        ...

    def add_dynamic_object(
        self,
        key: str,
        value: Any,
        name: Optional[str] = None,
        dim: Optional[int] = None,
        parse: bool = False,
        check_tokens: bool = True,
    ) -> None:
        """Registers *value* as the dynamic dimension named *key*.

        :param key: symbolic dimension name
        :param value: :class:`torch.SymInt`, :class:`int`, or similar
        :param name: tensor input this dimension originates from
        :param dim: axis of *name* this dimension corresponds to
        :param parse: also register sub-expressions of *value*
        :param check_tokens: verify sub-tokens are registered first
        """
        ...

    def make_new_dynamic_shape(self, rank: int, prefix: str = "d") -> Tuple[Any, ...]:
        """Creates a dynamic shape of the given *rank* with fresh symbolic
        dimensions.

        :param rank: number of dimensions
        :param prefix: prefix for the generated dimension names
        :return: tuple of :class:`torch.SymInt` objects
        """
        ...

    # ------------------------------------------------------------------
    # Sub-builder and local-function support
    # ------------------------------------------------------------------

    def make_nodes(
        self,
        builder: Any,
        input_names: List[str],
        output_names: List[str],
        prefix: str = "",
        function_options: Optional[Any] = None,
        optimize: bool = False,
        force_rename_with_prefix: Optional[str] = None,
    ) -> Any:
        """Appends all nodes and initializers from *builder* into this
        graph.

        :param builder: source :class:`GraphBuilder`
        :param input_names: names of the inputs that correspond to
            *builder*'s inputs in the current graph
        :param output_names: desired output names in the current graph
        :param prefix: prefix applied to every result name from *builder*
            when *function_options* is ``None``
        :param function_options: when set the sub-graph is exported as a
            local ONNX function
        :param optimize: run optimizations on the appended sub-graph
        :param force_rename_with_prefix: force this prefix regardless of
            *function_options*
        :return: output name(s) in the current graph
        """
        ...

    def make_local_function(
        self,
        builder: Any,
        function_options: Any,
        optimize: bool = False,
        metadata_props: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], Tuple[str, str]]:
        """Converts *builder* into a local ONNX function and registers it.

        :param builder: source :class:`GraphBuilder` for the function body
        :param function_options: controls naming, inlining, and weight
            handling
        :param optimize: run optimizations on the function body
        :param metadata_props: extra key/value metadata for the function
        :return: ``(added_initializers, (domain, name))``
        """
        ...

    def make_subset_builder(
        self,
        input_names: List[str],
        name: str,
        domain: str,
        add_local_functions: bool = False,
    ) -> Any:
        """Creates a reduced copy of this builder that covers only the
        tensors reachable from *input_names*.

        :param input_names: graph inputs for the subset
        :param name: local-function name for the subset
        :param domain: local-function domain for the subset
        :param add_local_functions: copy local functions into the subset
        :return: new :class:`GraphBuilder` instance
        """
        ...

    # ------------------------------------------------------------------
    # Miscellaneous methods
    # ------------------------------------------------------------------

    def add_stat(self, kind: str, name: str) -> None:
        """Records a conversion statistic (no-op in most implementations).

        :param kind: statistic category
        :param name: statistic name
        """
        ...

    def pretty_text(self, add_fx_graph: bool = False, recursive: bool = True) -> str:
        """Returns a human-readable multi-line description of the graph.

        :param add_fx_graph: include the original FX graph text
        :param recursive: include local functions
        :return: formatted string
        """
        ...

    def register_users(self, name: str, users: Iterable[str]) -> None:
        """Registers the consumers of tensor *name* (used for validation).

        :param name: producer tensor name
        :param users: iterable of consumer tensor names
        """
        ...

    def extract_input_names_from_args(self, args: Any) -> List[str]:
        """Extracts all known tensor names from a (possibly nested) args
        structure.

        :param args: flat or nested sequence of values; strings that are
            known graph names are collected
        :return: deduplicated list of input tensor names
        """
        ...
