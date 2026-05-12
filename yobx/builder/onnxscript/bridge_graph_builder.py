from __future__ import annotations
import contextlib
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Union, Tuple
import numpy as np
import onnx
from onnx import TensorProto
from onnx import numpy_helper as onnx_numpy_helper
from onnx.model_container import make_large_tensor_proto
import onnx_ir as ir
from ...typing import GraphBuilderExtendedProtocol, OpsetProtocol
from ...container import ExtendedModelContainer, ExportArtifact
from ...helpers.helper import size_type
from ...helpers.onnx_helper import _default_OPSET_TO_IR_VERSION
from ...xshape.shape_type_compute import set_type_shape_unary_op
from ...xshape._shape_helper import DYNAMIC_SHAPE


def to_ir_dtype(elem_type: Optional[int]) -> Optional[ir.DataType]:
    """
    Converts an ONNX ``TensorProto`` element-type integer to :class:`ir.DataType`.

    :param elem_type: ONNX element type (e.g. ``TensorProto.FLOAT == 1``),
        or ``None`` / 0 for *unknown*.
    :return: Corresponding :class:`ir.DataType`, or ``None`` when unknown.
    """
    if not elem_type:
        return None
    return ir.DataType(elem_type)


def to_ir_shape(shape: Optional[Sequence[Optional[Union[int, str]]]]) -> Optional[ir.Shape]:
    """
    Converts a yobx-style shape tuple to :class:`ir.Shape`.

    :param shape: A sequence of dimension sizes.  Each element may be an
        ``int`` (static), a ``str`` (symbolic / dynamic), or ``None``
        (fully unknown dimension).
    :return: :class:`ir.Shape`, or ``None`` when *shape* itself is ``None``.
    """
    if shape is None:
        return None
    return ir.Shape(list(shape))


def value_to_ir_tensor(value: Any, name: str) -> ir.TensorProtocol:
    """
    Converts an initializer *value* to an :class:`ir.TensorProtocol`.

    Supported input types:

    * :class:`numpy.ndarray`
    * scalar Python ``int`` or ``float`` (promoted to 0-D arrays)
    * :class:`onnx.TensorProto` (converted via :func:`ir.from_proto`)
    * Any object already satisfying :class:`ir.TensorProtocol`

    :param value: The raw initializer value.
    :param name: Name that should be associated with the resulting tensor.
    :return: An :class:`ir.TensorProtocol` suitable for passing to
        :meth:`onnxscript._internal.builder.GraphBuilder.initializer`.
    :raises TypeError: When *value* has an unsupported type.
    """
    if isinstance(value, onnx.TensorProto):
        t = ir.from_proto(value)
        return t  # type: ignore
    if isinstance(value, int):
        value = np.array(value, dtype=np.int64)
    elif isinstance(value, float):
        value = np.array(value, dtype=np.float32)
    if isinstance(value, np.ndarray):
        return ir.tensor(value, name=name)
    # Try to use it as-is (e.g. already an ir.TensorProtocol subclass)
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return ir.tensor(np.array(value), name=name)
    raise TypeError(
        f"Cannot convert initializer {name!r} of type {type(value)} "
        "to an onnx-ir tensor.  Supported types: numpy.ndarray, int, float, "
        "onnx.TensorProto."
    )


class OnnxScriptGraphBuilderOpset(OpsetProtocol):
    """Implements :class:`yobx.typing.OpsetProtocol`."""

    def __init__(self, builder):
        self.builder = builder

    def __getattr__(self, op_type: str) -> Callable[..., Union[str, Tuple[str, ...]]]:
        return partial(self.builder._make_node, op_type)


class OnnxScriptGraphBuilder(GraphBuilderExtendedProtocol):
    """
    Bridge builder that exposes a yobx-compatible API over onnxscript's IR.
    It takes onnxscript `GraphBuilder
    <https://github.com/microsoft/onnxscript/blob/main/onnxscript/_internal/builder.py#L104>`_
    implements the API :ref:`l-design-expected-api`.

    :param target_opset_or_existing_proto: Either a single opset version (``int``) or
        a mapping ``{domain: version}`` (``Dict[str, int]``).  For example
        ``18`` or ``{"": 18, "com.microsoft": 1}``.
    :param ir_version: ONNX IR version to use when exporting.  When ``None``
        a sensible default is chosen based on the main opset version.

    The builder wraps an :class:`onnxscript._internal.builder.GraphBuilder`
    and keeps a ``name → ir.Value`` registry so that callers can reference
    previously created tensors by their string name (as in the yobx API)
    rather than by ``ir.Value`` handle.

    Typical usage::

        gr = OnnxScriptGraphBuilder(18)
        gr.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        w_name = gr.make_initializer("W", np.eye(4, dtype=np.float32))
        (y_name,) = gr.make_node("MatMul", ["X", w_name], 1, name="mm")
        gr.make_tensor_output(y_name, TensorProto.FLOAT, (None, 4))
        proto = gr.to_onnx()
    """

    def __init__(
        self,
        target_opset_or_existing_proto: Union[int, Dict[str, int]],
        ir_version: Optional[int] = None,
    ) -> None:
        from onnxscript._internal.builder import GraphBuilder as OSGraphBuilder

        if isinstance(target_opset_or_existing_proto, (int, dict)):
            self.opsets = (
                {"": target_opset_or_existing_proto}
                if isinstance(target_opset_or_existing_proto, int)
                else target_opset_or_existing_proto
            )
        else:
            raise NotImplementedError(
                f"Type {type(target_opset_or_existing_proto)} is not supported."
            )

        self.ir_version = (
            ir_version if ir_version else _default_OPSET_TO_IR_VERSION()[self.main_opset]
        )
        assert self.ir_version, f"{self.ir_version=} is wrong, {self.main_opset=}"

        self._graph = ir.Graph(
            name="graph", inputs=[], outputs=[], nodes=[], opset_imports=self.opsets
        )
        self._inner: OSGraphBuilder = OSGraphBuilder(self._graph)

        # Mapping from the user-visible name → ir.Value
        self._name_to_value: Dict[str, ir.Value] = {}
        # Counter for auto-generating output names
        self._output_counter: int = 0
        self._prefix_stack: List[str] = []
        self._unique_names: Set[str] = set()
        self._op = OnnxScriptGraphBuilderOpset(self)

    @property
    def op(self) -> OpsetProtocol:
        """Returns the shortcut to OpsetProtocal."""
        return self._op

    @property
    def main_opset(self) -> int:
        "Returns the opset for the main domain (assuming it is used)."
        return self.opsets[""]

    def get_opset(self, domain: str, exc: bool = True) -> int:
        """
        Returns the opset version for a specific domain.

        :param domain: domain name
        :param exc: raise an exception if missing
        :return: version
        """
        if exc:
            assert (
                domain in self.opsets
            ), f"Domain {domain!r} is not registered in opsets={self.opsets!r}."
        return self.opsets.get(domain, 0)

    def set_opset(self, domain: str, version: int = 1) -> None:
        """
        Sets the opset version for a domain.
        Checks the version is the same if it already exists.

        :param domain: domain name to register
        :param version: opset version for the domain
        """
        if domain in self.opsets:
            assert version == self.opsets[domain], (
                f"Version mismatch for domain={domain!r}, current is "
                f"{self.opsets[domain]}, new is {version}."
            )
            return
        self.opsets[domain] = version
        # Keep the underlying graph's opset_imports in sync.
        self._graph.opset_imports[domain] = version

    def add_domain(self, domain: str, version: int = 1) -> None:
        """Deprecated. Use :meth:`set_opset` instead."""
        self.set_opset(domain, version)

    def has_opset(self, domain: str) -> int:
        """
        Returns the opset version for a domain, or 0 if the domain is not registered.

        :param domain: domain name
        :return: opset version, or ``0`` if the domain is unknown
        """
        return self.opsets.get(domain, 0)

    @property
    def inner_builder(self):
        """
        The underlying :class:`onnxscript._internal.builder.GraphBuilder`.
        Use this to access onnxscript-native functionality that is not
        exposed through the yobx-compatible bridge API.

        :return: :class:`onnxscript._internal.builder.GraphBuilder`
        """
        return self._inner

    # ------------------------------------------------------------------
    # Name registry helpers
    # ------------------------------------------------------------------

    def _make_node(
        self,
        op_type: str,
        *args,
        domain: str = "",
        outputs: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """Creates a node."""
        new_args = []
        for a in args:
            if isinstance(a, str):
                # Existing values are referenced by name; look up their ir.Value.
                new_args.append(self._name_to_value[a])
            elif a is None:
                # Represent missing optional inputs as None, not as an initializer.
                new_args.append(None)  # type: ignore
            else:
                # Create an initializer and pass its corresponding ir.Value to the op.
                init_name = self.make_initializer("", a)
                new_args.append(self._name_to_value[init_name])

        if "name" in kwargs:
            # name is not supported by onnxscript.GraphBuilder.
            kwargs.pop("name")

        # outputs= and domain= are proper call_op parameters; do not embed them in kwargs.
        outputs_spec: Union[int, List[str]] = list(outputs) if outputs is not None else 1
        output = self._inner.call_op(
            op_type, new_args, kwargs, domain=domain, outputs=outputs_spec
        )
        if isinstance(output, ir.Value):
            if outputs:
                assert len(outputs) == 1
                self._register(outputs[0], output)
                return outputs[0]
            name = self.unique_name(op_type)
            self._register(name, output)
            return name
        res = []
        for i, o in enumerate(output):
            assert isinstance(o, ir.Value), (
                f"This should be an ir.Value not {type(o)} for operator "
                f"{op_type!r} called with args={new_args}"
            )
            if outputs:
                n = outputs[i]
                self._register(n, o)
                res.append(n)
            else:
                n = self.unique_name(op_type)
                self._register(n, o)
                res.append(n)
        return tuple(res)

    def unique_name(self, prefix: str) -> str:
        """Returns a unique name."""
        if self._prefix_stack:
            prefix = f"{'__'.join(self._prefix_stack)}__{prefix}"
        if prefix in self._unique_names:
            i = 2
            sug = f"{prefix}2"
            while sug in self._unique_names:
                i += 1
                sug = f"{prefix}{i}"
            self._unique_names.add(sug)
            return sug
        self._unique_names.add(prefix)
        return prefix

    @contextlib.contextmanager
    def prefix_name_context(self, prefix: str) -> Generator:
        """Context manager that pushes *prefix* onto the prefix stack.

        While active, :meth:`unique_name` prepends the joined stack to every
        requested name so that all tensor names produced inside the block are
        scoped to the current context.  The prefix is removed when the block
        exits, even if an exception is raised.

        :param prefix: scope prefix to push (e.g. a pipeline step name)
        """
        self._prefix_stack.append(prefix)
        try:
            yield
        finally:
            self._prefix_stack.pop()

    def has_name(self, name: str) -> bool:
        """
        Returns ``True`` when *name* is a known value in this graph.

        :param name: Tensor name to query.
        """
        return name in self._name_to_value

    def has_type(self, name: str) -> int:
        """Tells if a value has a type."""
        if name not in self._name_to_value:
            return False
        value = self._name_to_value[name]
        dtype = value.type
        if not dtype:
            return False
        return True

    def get_type(self, name: str) -> int:
        """Returns the type."""
        if name not in self._name_to_value:
            return False
        value = self._name_to_value[name]
        dtype = value.type
        if not dtype:
            return 0
        return int(dtype.elem_type)  # type: ignore

    def set_type(self, name: str, itype: int):
        """Sets the type."""
        if name not in self._name_to_value:
            return False
        value = self._name_to_value[name]
        value.type = ir.TensorType(ir.DataType(itype))

    def has_shape(self, name: str) -> bool:
        """Tells if a value has a shape."""
        if name not in self._name_to_value:
            return False
        value = self._name_to_value[name]
        shape = value.shape
        if shape is None:
            return False
        return True

    def get_shape(self, name: str) -> DYNAMIC_SHAPE:
        """Returns the shape."""
        assert name in self._name_to_value, f"Name {name!r} is not registered."
        value = self._name_to_value[name]
        assert value is not None, f"Name {name!r} has a shape but it is None."
        # A dynamic dimension is a ir.SymbolicDim.
        return tuple(s if isinstance(s, (int, str)) else s.value for s in value.shape)  # type: ignore

    def set_shape(self, name: str, shape: DYNAMIC_SHAPE, allow_zero: bool = False):
        """Sets the shape."""
        assert shape is not None, f"shape cannot be empty for name={name!r}"
        if name not in self._name_to_value:
            return False
        value = self._name_to_value[name]
        value.shape = ir.Shape(shape)
        assert (
            allow_zero or not shape or 0 not in shape
        ), f"Shape {shape} for name={name!r} is null."

    def has_device(self, name: str) -> int:
        """
        Tells if a value has a device.
        This is not supported right now.
        """
        return False

    def get_device(self, name: str) -> int:
        """Returns the device. This is not supported right now."""
        raise NotImplementedError(
            f"device for {name!r} is not available with builder {self.__class__}"
        )

    def onnx_dtype_to_np_dtype(self, itype: int) -> np.dtype:
        """See :func:`yobx.helpers.onnx_helper.tensor_dtype_to_np_dtype`."""
        from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

        return tensor_dtype_to_np_dtype(itype)

    def get_value(self, name: str) -> ir.Value:
        """
        Returns the :class:`ir.Value` associated with *name*.

        :param name: Tensor name.
        :raises KeyError: When *name* has not been registered.
        """
        try:
            return self._name_to_value[name]
        except KeyError:
            raise KeyError(
                f"Name {name!r} is not known. Known names: {sorted(self._name_to_value)}"
            ) from None

    def set_type_shape_unary_op(
        self, name: str, input_name: str, itype: Optional[int] = None
    ) -> bool:
        return set_type_shape_unary_op(self, name, input_name, itype)  # type: ignore[arg-type]

    def _register(self, name: str, value: ir.Value) -> None:
        """Register *value* under *name* in the internal name registry."""
        assert isinstance(value, ir.Value), f"Unexpected type {type(value)} for name={name!r}"
        value.name = name
        self._name_to_value[name] = value

    # ------------------------------------------------------------------
    # Core builder API (mirrors yobx GraphBuilder)
    # ------------------------------------------------------------------

    def make_tensor_input(
        self,
        name: str,
        elem_type: Optional[int] = None,
        shape: Optional[Sequence[Optional[Union[int, str]]]] = None,
        device: Optional[int] = None,
    ) -> str:
        """
        Adds a graph input and return its name.

        :param name: Input tensor name.
        :param elem_type: ONNX element type (e.g. ``TensorProto.FLOAT``).
            Pass ``None`` or ``0`` if unknown (only valid for function graphs).
        :param shape: Tensor shape.  Use ``None`` for a fully unknown
            dimension and a ``str`` for a symbolic / dynamic dimension.
        :param device: unused for the time being
        :return: The registered name (same as *name*).
        """
        dtype = to_ir_dtype(elem_type)
        ir_shape = to_ir_shape(shape)

        tensor_type: Optional[ir.TensorType] = ir.TensorType(dtype) if dtype is not None else None
        value = ir.Value(name=name, type=tensor_type, shape=ir_shape)
        self._graph.inputs.append(value)
        self._register(name, value)
        return name

    def make_tensor_output(
        self,
        name: Union[str, List[str]],
        elem_type: Optional[int] = None,
        shape: Optional[Sequence[Optional[Union[int, str]]]] = None,
        indexed: bool = False,
        allow_untyped_output: bool = False,
    ) -> Union[str, List[str]]:
        """
        Registers an existing value as a graph output and return its name.

        :param name: Name (or list of names) of the tensor(s) to mark as
            graph output(s).  Must already exist in this builder (i.e. have
            been created by :meth:`make_tensor_input`,
            :meth:`make_initializer`, or :meth:`make_node`).
        :param elem_type: Optional element type hint; used to set the type on
            the ``ir.Value`` if it was not already inferred.
        :param shape: Optional shape hint; used to set / refine the shape on
            the ``ir.Value`` if not already inferred.
        :param indexed: unused
        :param allow_untyped_output: allows output with no shape and/or no type
        :return: The name (or list of names), matching the *name* argument.
        """
        if isinstance(name, list):
            assert all(isinstance(n, str) for n in name)  # type happiness
            res = [self.make_tensor_output(n, elem_type=elem_type, shape=shape) for n in name]
            assert all(isinstance(n, str) for n in res)  # type happiness
            return res  # type: ignore

        value = self.get_value(name)

        # Optionally apply type/shape hints
        dtype = to_ir_dtype(elem_type)
        if dtype is not None and value.type is None:
            value.type = ir.TensorType(dtype)

        ir_shape = to_ir_shape(shape)
        if ir_shape is not None and value.shape is None:
            value.shape = ir_shape

        self._graph.outputs.append(value)
        return name

    def make_initializer(self, name: str, value: Any) -> str:
        """
        Adds an initializer tensor and return its name.

        :param name: Name for the initializer.  May be an empty string ``""``
            in which case a unique name is generated automatically.
        :param value: Initializer data.  Supported types: :class:`numpy.ndarray`,
            scalar ``int``/``float``, :class:`onnx.TensorProto`.
        :return: The final registered name (may differ from *name* when *name*
            is empty).
        """
        if not name:
            name = self.unique_name("init_")

        tensor = value_to_ir_tensor(value, name)
        ir_value = self._inner.initializer(tensor, name=name, qualify=False)
        self._register(name, ir_value)
        return name

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, str, List[str]] = 1,
        domain: str = "",
        attributes: Optional[List[onnx.AttributeProto]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, ...]]:
        """
        Creates an ONNX node and return its output name(s).

        :param op_type: ONNX operator type (e.g. ``"Relu"``, ``"MatMul"``).
        :param inputs: Input tensor name(s).  Each name must have been
            registered previously via :meth:`make_tensor_input`,
            :meth:`make_initializer`, or a previous :meth:`make_node` call.
            Pass an empty string ``""`` for optional absent inputs.
        :param outputs: Either an ``int`` (number of outputs, names are
            auto-generated), a single output ``str``, or a list of output
            ``str`` names.
        :param domain: Operator domain (default: ``""`` = standard ONNX).
        :param attributes: Additional :class:`onnx.AttributeProto` instances
            to attach to the node (supplementary to *kwargs*).
        :param name: Optional node name (for debugging / profiling).
        :param kwargs: Operator attributes as Python primitives (``int``,
            ``float``, ``str``, ``List[int]``, …).
        :return: The output name when a single output is created, or a list
            of names when multiple outputs are created.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        # Resolve input names → ir.Value objects
        ir_inputs: List[Optional[ir.Value]] = []
        for inp in inputs:
            if inp == "":
                ir_inputs.append(None)
            else:
                ir_inputs.append(self.get_value(inp))

        # Build output specification
        if isinstance(outputs, int):
            output_count = outputs
            output_names: Optional[List[str]] = None
        elif isinstance(outputs, str):
            output_count = 1
            output_names = [outputs]
        else:
            output_count = len(outputs)
            output_names = list(outputs)

        # Convert AttributeProto list to kwargs entries
        extra_kwargs: Dict[str, Any] = {}
        if attributes:
            for attr in attributes:
                ir_attr = ir.from_proto(attr)
                extra_kwargs[attr.name] = ir_attr.value  # type: ignore

        all_kwargs = {**extra_kwargs, **kwargs}
        # outputs= and domain= are proper call_op parameters (new onnxscript API);
        # do not embed them in the attributes dict to avoid spurious ONNX attributes.
        outputs_spec: Union[int, List[str]] = (
            output_names if output_names is not None else output_count
        )

        result = self._inner.call_op(
            op_type, ir_inputs, all_kwargs, domain=domain, outputs=outputs_spec  # type: ignore
        )

        if isinstance(result, ir.Value):
            result_list: List[ir.Value] = [result]
        else:
            result_list = list(result)

        # Determine the final user-visible names and register them.
        # When output names were explicitly provided, rename the ir.Value
        # objects so that the exported proto uses the requested names.
        if output_names is not None:
            final_names = output_names
            for n, v in zip(final_names, result_list):
                v.name = n
        else:
            # Use the actual ir.Value names as the user-visible keys
            final_names = []
            for v in result_list:
                final_names.append(v.name or f"tmp_{self._output_counter}")
                if not v.name:
                    self._output_counter += 1

        for n, v in zip(final_names, result_list):
            self._register(n, v)

        if len(final_names) == 1:
            return final_names[0]
        return tuple(final_names)

    @property
    def input_names(self) -> List[str]:
        """Returns input names."""
        return [i.name or "" for i in self._graph.inputs]

    @property
    def output_names(self) -> List[str]:
        """Returns output names."""
        return [i.name or "" for i in self._graph.outputs]

    def get_debug_msg(self) -> str:
        """Returns information useful for understanding where an error could come from.

        :return: a summary of the current graph state as a string
        """
        lines = [
            f"OnnxScriptGraphBuilder opsets={self.opsets!r}",
            f"  inputs:  {self.input_names}",
            f"  outputs: {self.output_names}",
            f"  initializers: {list(self._name_to_value.keys())}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_onnx(
        self, large_model: bool = False, external_threshold: int = 1024, inline: bool = True
    ) -> ExportArtifact:
        """Exports the graph as an ONNX :class:`~onnx.ModelProto`.

        :param large_model: if True returns a :class:`onnx.model_container.ModelContainer`,
            it lets the user to decide later if the weights should be part of the model
            or saved as external weights
        :param external_threshold: if large_model is True, every tensor above this limit
            (in bytes) is stored as external
        :param inline: inline local function it any (this is currently not used)
        :return: artifact
        """
        ir_model = ir.Model(self._graph, ir_version=self.ir_version)
        proto = ir.to_proto(ir_model)

        # onnx >= 1.20 requires the ``shape`` field to be present even when
        # the shape is fully unknown.  Add an empty TensorShapeProto where
        # needed (mirrors the fix in yobx/builder/light/_graph.py).
        for value_info in list(proto.graph.input) + list(proto.graph.output):
            if value_info.type.HasField(
                "tensor_type"
            ) and not value_info.type.tensor_type.HasField("shape"):
                value_info.type.tensor_type.shape.CopyFrom(onnx.TensorShapeProto())

        if not large_model:
            return ExportArtifact(proto=proto)

        # Extract initializers that exceed the threshold into an ExtendedModelContainer.
        large_initializers: Dict[str, np.ndarray] = {}
        new_initializers = []
        for init in proto.graph.initializer:
            if init.data_type == TensorProto.STRING:
                # String initializers have no fixed binary size; never externalize them.
                new_initializers.append(init)
                continue
            size = int(np.prod(init.dims) if init.dims else 1) * size_type(init.data_type)
            if size >= external_threshold:
                # The location must start with '#' so that onnx's check_model
                # skips it (it is treated as an in-memory external reference).
                location = f"#{init.name}"
                arr = onnx_numpy_helper.to_array(init)
                large_tensor = make_large_tensor_proto(
                    location, init.name, init.data_type, list(init.dims)
                )
                new_initializers.append(large_tensor)
                large_initializers[location] = arr
            else:
                new_initializers.append(init)

        del proto.graph.initializer[:]
        proto.graph.initializer.extend(new_initializers)

        lm = ExtendedModelContainer()
        lm.model_proto = proto
        if large_initializers:
            lm.set_large_initializers(large_initializers)
            lm.check_large_initializers()
        return ExportArtifact(container=lm)
