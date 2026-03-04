"""
Bridge between yobx :class:`~yobx.xbuilder.GraphBuilder` API and
:class:`onnxscript._internal.builder.GraphBuilder`.

The :class:`OnnxScriptGraphBuilder` wraps the onnxscript imperative IR builder
and exposes a simplified interface that mirrors the most commonly used methods
of the yobx ``GraphBuilder``.  This allows callers to build ONNX graphs using
the familiar yobx naming conventions (string-based inputs/outputs, ONNX
``TensorProto`` element-type integers, tuple shapes) while delegating the
heavy-lifting to onnxscript's native IR layer.

Example::

    import numpy as np
    from onnx import TensorProto
    from yobx.builder.onnxscript_bridge import OnnxScriptGraphBuilder

    gr = OnnxScriptGraphBuilder({"": 18})
    gr.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
    init_name = gr.make_initializer("W", np.ones((4, 2), dtype=np.float32))
    out_names = gr.make_node("MatMul", ["X", init_name], 1, name="mm")
    gr.make_tensor_output(out_names, TensorProto.FLOAT, (None, 2))
    model_proto = gr.to_onnx()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from onnx import AttributeProto, ModelProto, TensorProto

import onnx_ir as ir


def _to_ir_dtype(elem_type: Optional[int]) -> Optional[ir.DataType]:
    """Convert an ONNX ``TensorProto`` element-type integer to :class:`ir.DataType`.

    :param elem_type: ONNX element type (e.g. ``TensorProto.FLOAT == 1``),
        or ``None`` / 0 for *unknown*.
    :return: Corresponding :class:`ir.DataType`, or ``None`` when unknown.
    """
    if not elem_type:
        return None
    return ir.DataType(elem_type)


def _to_ir_shape(
    shape: Optional[Sequence[Optional[Union[int, str]]]]
) -> Optional[ir.Shape]:
    """Convert a yobx-style shape tuple to :class:`ir.Shape`.

    :param shape: A sequence of dimension sizes.  Each element may be an
        ``int`` (static), a ``str`` (symbolic / dynamic), or ``None``
        (fully unknown dimension).
    :return: :class:`ir.Shape`, or ``None`` when *shape* itself is ``None``.
    """
    if shape is None:
        return None
    return ir.Shape(list(shape))


def _value_to_ir_tensor(value: Any, name: str) -> ir.TensorProtocol:
    """Convert an initializer *value* to an :class:`ir.TensorProtocol`.

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
    if isinstance(value, TensorProto):
        t = ir.from_proto(value)
        return t
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


def _kwargs_to_ir_attrs(
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a mixed dict of attribute values to a form accepted by
    :meth:`onnxscript._internal.builder.GraphBuilder.call_op`.

    :class:`onnx.AttributeProto` instances are converted to their Python
    equivalents via :func:`ir.from_proto`; all other values are passed
    through unchanged.

    :param kwargs: Raw keyword-argument dict as produced by yobx
        :meth:`~yobx.xbuilder.GraphBuilder.make_node`.
    :return: Filtered/converted dict suitable for ``call_op``.
    """
    result: Dict[str, Any] = {}
    for key, val in kwargs.items():
        if isinstance(val, AttributeProto):
            ir_attr = ir.from_proto(val)
            result[key] = ir_attr.value
        else:
            result[key] = val
    return result


class OnnxScriptGraphBuilder:
    """Bridge builder that exposes a yobx-compatible API over onnxscript's IR.

    :param target_opset_or_opsets: Either a single opset version (``int``) or
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
        target_opset_or_opsets: Union[int, Dict[str, int]],
        ir_version: Optional[int] = None,
    ) -> None:
        from onnxscript._internal.builder import GraphBuilder as _OSGraphBuilder

        if isinstance(target_opset_or_opsets, int):
            opsets: Dict[str, int] = {"": target_opset_or_opsets}
        else:
            opsets = dict(target_opset_or_opsets)

        self._opsets = opsets
        self._ir_version = ir_version

        self._graph = ir.Graph(
            name="graph",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=opsets,
        )
        self._inner: _OSGraphBuilder = _OSGraphBuilder(self._graph)

        # Mapping from the user-visible name → ir.Value
        self._name_to_value: Dict[str, ir.Value] = {}
        # Counter for auto-generating output names
        self._output_counter: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def inner_builder(self):
        """The underlying :class:`onnxscript._internal.builder.GraphBuilder`.

        Use this to access onnxscript-native functionality that is not
        exposed through the yobx-compatible bridge API.

        :return: :class:`onnxscript._internal.builder.GraphBuilder`
        """
        return self._inner

    @property
    def op(self):
        """Dynamic op dispatcher from the underlying onnxscript builder.

        Equivalent to ``self.inner_builder.op``.  Allows constructs such as::

            y = gr.op.Relu(x_value)

        where *x_value* is an :class:`ir.Value` retrieved from
        :meth:`get_value`.
        """
        return self._inner.op

    @property
    def opsets(self) -> Dict[str, int]:
        """Opset dictionary ``{domain: version}``."""
        return dict(self._opsets)

    # ------------------------------------------------------------------
    # Name registry helpers
    # ------------------------------------------------------------------

    def has_name(self, name: str) -> bool:
        """Return ``True`` when *name* is a known value in this graph.

        :param name: Tensor name to query.
        """
        return name in self._name_to_value

    def get_value(self, name: str) -> ir.Value:
        """Return the :class:`ir.Value` associated with *name*.

        :param name: Tensor name.
        :raises KeyError: When *name* has not been registered.
        """
        try:
            return self._name_to_value[name]
        except KeyError:
            raise KeyError(
                f"Name {name!r} is not known.  "
                f"Known names: {sorted(self._name_to_value)}"
            ) from None

    def _register(self, name: str, value: ir.Value) -> None:
        """Register *value* under *name* in the internal name registry."""
        self._name_to_value[name] = value

    # ------------------------------------------------------------------
    # Core builder API (mirrors yobx GraphBuilder)
    # ------------------------------------------------------------------

    def make_tensor_input(
        self,
        name: str,
        elem_type: Optional[int] = None,
        shape: Optional[Sequence[Optional[Union[int, str]]]] = None,
    ) -> str:
        """Add a graph input and return its name.

        :param name: Input tensor name.
        :param elem_type: ONNX element type (e.g. ``TensorProto.FLOAT``).
            Pass ``None`` or ``0`` if unknown (only valid for function graphs).
        :param shape: Tensor shape.  Use ``None`` for a fully unknown
            dimension and a ``str`` for a symbolic / dynamic dimension.
        :return: The registered name (same as *name*).
        """
        dtype = _to_ir_dtype(elem_type)
        ir_shape = _to_ir_shape(shape)

        tensor_type: Optional[ir.TensorType] = (
            ir.TensorType(dtype) if dtype is not None else None
        )
        value = ir.Value(name=name, type=tensor_type, shape=ir_shape)
        self._graph.inputs.append(value)
        self._register(name, value)
        return name

    def make_tensor_output(
        self,
        name: Union[str, List[str]],
        elem_type: Optional[int] = None,
        shape: Optional[Sequence[Optional[Union[int, str]]]] = None,
    ) -> Union[str, List[str]]:
        """Register an existing value as a graph output and return its name.

        :param name: Name (or list of names) of the tensor(s) to mark as
            graph output(s).  Must already exist in this builder (i.e. have
            been created by :meth:`make_tensor_input`,
            :meth:`make_initializer`, or :meth:`make_node`).
        :param elem_type: Optional element type hint; used to set the type on
            the ``ir.Value`` if it was not already inferred.
        :param shape: Optional shape hint; used to set / refine the shape on
            the ``ir.Value`` if not already inferred.
        :return: The name (or list of names), matching the *name* argument.
        """
        if isinstance(name, list):
            return [
                self.make_tensor_output(n, elem_type=elem_type, shape=shape)  # type: ignore[misc]
                for n in name
            ]

        value = self.get_value(name)

        # Optionally apply type/shape hints
        dtype = _to_ir_dtype(elem_type)
        if dtype is not None and value.type is None:
            value.type = ir.TensorType(dtype)

        ir_shape = _to_ir_shape(shape)
        if ir_shape is not None and value.shape is None:
            value.shape = ir_shape

        self._graph.outputs.append(value)
        return name

    def make_initializer(
        self,
        name: str,
        value: Any,
    ) -> str:
        """Add an initializer tensor and return its name.

        :param name: Name for the initializer.  May be an empty string ``""``
            in which case a unique name is generated automatically.
        :param value: Initializer data.  Supported types: :class:`numpy.ndarray`,
            scalar ``int``/``float``, :class:`onnx.TensorProto`.
        :return: The final registered name (may differ from *name* when *name*
            is empty).
        """
        if not name:
            name = f"init_{len(self._name_to_value)}"

        tensor = _value_to_ir_tensor(value, name)
        ir_value = self._inner.initializer(tensor, name=name, qualify=False)
        self._register(name, ir_value)
        return name

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, str, List[str]] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Create an ONNX node and return its output name(s).

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
                extra_kwargs[attr.name] = ir_attr.value

        # Merge all attribute sources; user kwargs take precedence
        all_kwargs = _kwargs_to_ir_attrs({**extra_kwargs, **kwargs})

        # Pass output specification to call_op
        if output_names is not None:
            all_kwargs["_outputs"] = output_names
        else:
            all_kwargs["_outputs"] = output_count

        if domain:
            all_kwargs["_domain"] = domain

        # Call onnxscript builder
        result = self._inner.call_op(op_type, ir_inputs, all_kwargs)

        # Normalise to list
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
        return final_names

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_onnx(self, ir_version: Optional[int] = None) -> ModelProto:
        """Export the graph as an ONNX :class:`~onnx.ModelProto`.

        :param ir_version: Override the IR version for this call.  Falls back
            to the value passed at construction time, or an automatic default.
        :return: A fully populated :class:`~onnx.ModelProto`.
        """
        from onnx import TensorShapeProto

        effective_ir_version = ir_version or self._ir_version
        if effective_ir_version is None:
            # Pick a reasonable default based on the main opset version
            main_opset = self._opsets.get("", 18)
            effective_ir_version = _default_ir_version(main_opset)

        ir_model = ir.Model(
            self._graph,
            ir_version=effective_ir_version,
        )
        proto = ir.to_proto(ir_model)

        # onnx >= 1.20 requires the ``shape`` field to be present even when
        # the shape is fully unknown.  Add an empty TensorShapeProto where
        # needed (mirrors the fix in yobx/builder/light/_graph.py).
        for value_info in list(proto.graph.input) + list(proto.graph.output):
            if (
                value_info.type.HasField("tensor_type")
                and not value_info.type.tensor_type.HasField("shape")
            ):
                value_info.type.tensor_type.shape.CopyFrom(TensorShapeProto())

        return proto


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_ir_version(opset_version: int) -> int:
    """Return a sensible ONNX IR version for a given opset version.

    :param opset_version: ONNX opset version.
    :return: Recommended IR version.
    """
    # Mapping taken from the ONNX spec
    _OPSET_TO_IR: Dict[int, int] = {
        1: 3,
        2: 3,
        3: 3,
        4: 3,
        5: 3,
        6: 3,
        7: 3,
        8: 4,
        9: 4,
        10: 5,
        11: 6,
        12: 7,
        13: 7,
        14: 7,
        15: 8,
        16: 8,
        17: 8,
        18: 8,
        19: 9,
        20: 9,
        21: 10,
        22: 10,
    }
    return _OPSET_TO_IR.get(opset_version, 10)
