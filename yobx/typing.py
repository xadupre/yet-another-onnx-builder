from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Sequence,
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
class ConvertOptionsProtocol(Protocol):
    """Protocol for a class giving indications on how to convert a model."""

    def available_options(self) -> Sequence[str]:
        """Returns the list of options."""
        ...

    def has(self, option_name: str, piece: object, name: Optional[str] = None) -> bool:
        """Returns true if option `option_name` applies to `piece`"""
        ...


class DefaultConvertOptions(ConvertOptionsProtocol):
    """All options are disabled."""

    def available_options(self) -> Sequence[str]:
        """Returns the list of options."""
        return []

    def has(self, option_name: str, piece: object, name: Optional[str] = None) -> bool:
        """Returns always False."""
        return False


@runtime_checkable
class ExportArtifactProtocol(Protocol):
    """Protocol for ExportArtifact."""

    def save(self, file_path: str, all_tensors_to_one_file: bool = True):
        """Save the exported model to *file_path*.

        When a :class:`~yobx.container.ExtendedModelContainer` is present
        (``large_model=True`` was used during export) the model and its
        external weight files are saved via
        :meth:`~yobx.container.ExtendedModelContainer.save`.  Otherwise
        the proto is saved with :func:`onnx.save_model`.

        :param file_path: destination file path (including ``.onnx``
            extension).
        :param all_tensors_to_one_file: when saving a large model, write
            all external tensors into a single companion data file.
        """
        ...

    def get_proto(self, include_weights: bool = True) -> Any:
        """Return the ONNX proto, optionally with all weights inlined.

        When the export was performed with ``large_model=True`` (i.e.
        :attr:`container` is set), the raw :attr:`proto` has
        *external-data* placeholders instead of embedded weight tensors.
        Passing ``include_weights=True`` (the default) uses
        :meth:`~yobx.container.ExtendedModelContainer.to_ir` to build a
        fully self-contained :class:`~onnx.ModelProto`.

        :param include_weights: when ``True`` (default) embed the large
            initializers stored in :attr:`container` into the returned
            proto.  When ``False`` return the raw proto as-is.
        :return: :class:`~onnx.ModelProto`,
            :class:`~onnx.FunctionProto`, or
            :class:`~onnx.GraphProto`.

        Example::

            artifact = to_onnx(estimator, (X,), large_model=True)
            # Fully self-contained proto (weights embedded):
            proto_with_weights = artifact.get_proto(include_weights=True)
            # Proto with external-data placeholders:
            proto_no_weights = artifact.get_proto(include_weights=False)
        """
        ...

    @classmethod
    def load(
        cls, file_path: str, load_large_initializers: bool = True
    ) -> "ExportArtifactProtocol":
        """Load a saved model from *file_path*.

        If the file references external data (i.e. the model was saved
        with ``large_model=True``) an
        :class:`~yobx.container.ExtendedModelContainer` is created and
        returned in :attr:`container`.  Otherwise the proto is loaded
        directly with :func:`onnx.load` and :attr:`container` is ``None``.

        :param file_path: path to the ``.onnx`` file.
        :param load_large_initializers: when ``True`` (default) also load
            the large initializers stored alongside the model file.
        :return: :class:`ExportArtifact` with :attr:`filename` set to
            *file_path*.

        Example::

            artifact = ExportArtifact.load("model.onnx")
            proto = artifact.get_proto()
        """
        ...


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

    @property
    def convert_options(self) -> ConvertOptionsProtocol:
        """Returns converting options."""
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

    def make_initializer(
        self, name: str, value: Any, give_unique_name: bool = True, source: Optional[str] = None
    ) -> str:
        """Adds a constant initializer and returns its name.

        :param name: initializer name; may be empty to auto-generate a unique name
        :param value: initializer value (:class:`numpy.ndarray`,
            :class:`onnx.TensorProto`, ``int``, or ``float``)
        :param give_unique_name: changes the name if it is already taken, otherwise,
            the user should expect an exception to raised
        :param source: used to track where this initializer was added
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

    def to_onnx(self) -> ExportArtifactProtocol:
        """Exports the graph and returns an ONNX proto or model container.

        :return: a :class:`~onnx.ModelProto`, :class:`~onnx.GraphProto`,
            :class:`~onnx.FunctionProto`, or a model container object
        """
        ...

    def prefix_name_context(self, prefix: str) -> ContextManager[None]:
        """Context manager that scopes all :meth:`unique_name` calls to *prefix*.

        While the context is active, every name returned by :meth:`unique_name`
        is prefixed with the joined stack of active prefixes so that tensors
        produced inside the block are clearly associated with their enclosing
        scope (e.g. a pipeline step name).  Contexts may be nested; each level
        pushes one entry onto the stack.

        :param prefix: scope prefix to push (e.g. a pipeline step name)
        :return: a context manager; use as ``with g.prefix_name_context("step"): ...``
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

    def is_constant(self, name: str) -> bool:
        """Returns ``True`` when *name* is a known constant in the graph.

        :param name: result name
        :return: ``True`` if the result is a constant
        """
        ...

    def get_constant(
        self,
        name: str,
        exc: bool = True,
        computed_value: bool = False,
        as_shape: bool = False,
        multiple_outputs: bool = False,
    ) -> Optional[Any]:
        """Returns the constant value for *name*.

        :param name: constant name
        :param exc: raise an exception if the constant cannot be retrieved
        :param computed_value: if ``True``, evaluate ``NodeProto`` constants
            using a reference runtime
        :param as_shape: if ``True``, return a tuple of integers (a shape)
        :param multiple_outputs: allow the constant to have multiple outputs
        :return: the constant value, or ``None`` when *exc* is ``False`` and
            the value cannot be determined
        """
        ...

    def get_debug_msg(self) -> str:
        """Returns any information useful to understand where an error
        could come from. This message is expected to be part of any
        exception raised while converting a model.

        :return: information in a string
        """
        ...

    def value_as_shape(self, name: str) -> Any:
        """Returns the value of a result if it is known to represent a shape.

        A *shape value* is a 1-D ``INT64`` tensor (or symbolic equivalent)
        whose contents describe the dimensions of another tensor.  The method
        returns:

        * A ``tuple`` of ``int`` / symbolic-dimension values when the shape
          value is fully known (e.g. a constant or a previously recorded
          ``set_value_shape`` call).
        * ``None`` when the value cannot be determined.

        :param name: result name to query
        :return: tuple of dimension values, or ``None``
        """
        ...


@runtime_checkable
class GraphBuilderTorchProtocol(GraphBuilderExtendedProtocol, Protocol):
    """Protocol for graph builders that support the full torch-exporter API.

    This protocol extends :class:`GraphBuilderExtendedProtocol` with the
    additional methods and attributes used by
    :class:`~yobx.torch.interpreter.FxGraphInterpreter` (the *torch exporter*)
    when translating a ``torch.fx`` graph into ONNX.

    The extra surface covers:

    * **Rank helpers** — :meth:`has_rank`, :meth:`get_rank`, :meth:`set_rank`.
    * **Device helpers** — :meth:`has_device`, :meth:`get_device`,
      :meth:`set_device`.
    * **Extended type / shape** — :meth:`get_type_known`,
      :meth:`set_shapes_types`.
    * **Tensor-sequence support** — :meth:`make_tensor_sequence_input`.
    * **Dynamic-shape helpers** — :meth:`is_dynamic_shape`,
      :meth:`get_input_dynamic_shape`,
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
        self, name: str, device: Any, exc: bool = True, keep_this_device: bool = False
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

    def make_tensor_sequence_input(
        self, name: str, elem_type: Any, shape: Any, marker: str = ""
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

    def verify_dynamic_shape(
        self, shape: Any, name: Optional[str] = None, add: bool = True
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
        self, input_names: List[str], name: str, domain: str, add_local_functions: bool = False
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

    def get_constant_shape(self, name: str, exc: bool = True) -> Optional[Tuple[int, ...]]:
        """Returns the shape of constant *name*.

        :param name: result name
        :param exc: raise an exception if the shape cannot be determined
        :return: shape tuple, or ``None`` when *exc* is ``False``
        """
        ...

    def get_computed_constant(self, name: str, statistics: Optional[List[str]] = None) -> Any:
        """Returns the evaluated value of constant *name*.

        :param name: result name
        :param statistics: optional list of summary statistics to compute
            (``"min"``, ``"max"``); when given, a list of values is returned
        :return: :class:`numpy.ndarray` or a list of statistics
        """
        ...

    def get_constant_scalar(self, name: str, broadcast: bool = False) -> Union[int, float]:
        """Returns the scalar value of constant *name*.

        :param name: result name
        :param broadcast: accept shapes such as ``(1,)`` or ``(1,1)``
        :return: ``int``, ``float``, or ``complex``
        """
        ...

    def get_constant_or_attribute(
        self, node: Any, attribute: str, input_index: int, cvt: Optional[Callable] = None
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

    def get_attribute(self, node: Any, att_name: str, exc: bool = True) -> Optional[Any]:
        """Returns the :class:`~onnx.AttributeProto` named *att_name* on
        *node*.

        :param node: :class:`~onnx.NodeProto`
        :param att_name: attribute name
        :param exc: raise an exception if the attribute is missing
        :return: :class:`~onnx.AttributeProto` or ``None``
        """
        ...

    def get_attribute_with_default(self, node: Any, name: str, default_value: Any) -> Any:
        """Returns the value of attribute *name* on *node*, or
        *default_value* if the attribute is absent.

        :param node: :class:`~onnx.NodeProto`
        :param name: attribute name
        :param default_value: fallback value
        :return: attribute value or *default_value*
        """
        ...

    def get_attributes_with_default(self, node: Any, **default_values: Any) -> Dict[str, Any]:
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

    def has_exact_same_constant_in_context(self, name: str) -> Optional[bool]:
        """Checks whether an identical constant already exists in the graph.

        :param name: constant name to look up
        :return: ``True``/``False`` when known, ``None`` otherwise
        """
        ...

    def do_not_turn_constant_initializers_maybe_because_of_showing(self, name: str) -> bool:
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
        give_unique_name: bool = True,
    ) -> str:
        """Adds a constant initializer and returns its (possibly
        auto-generated) name.

        :param name: desired name (may be empty for auto-naming)
        :param value: :class:`numpy.ndarray` or
            :class:`~onnx.TensorProto` value
        :param external: store as external data
        :param msg: optional debug message
        :param source: optional source tag for debugging
        :param give_unique_name: generate a unique name when *name* is already
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

    def pretty_text(self, add_fx_graph: bool = False, recursive: bool = True) -> str:
        """Returns a human-readable text rendering of the graph.

        :param add_fx_graph: include the FX graph representation when
            available
        :param recursive: recurse into sub-graphs
        :return: multi-line string
        """
        ...
