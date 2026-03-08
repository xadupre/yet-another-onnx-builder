from __future__ import annotations

import collections
import importlib
import inspect
import typing
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import onnx
import spox

from ...helpers.onnx_helper import attr_proto_to_python, tensor_dtype_to_np_dtype
from ...typing import GraphBuilderExtendedProtocol, OpsetProtocol
from ...xshape._shape_helper import DYNAMIC_SHAPE
from ...xshape.shape_type_compute import set_type_shape_unary_op

# ---------------------------------------------------------------------------
# Opset module resolution
# ---------------------------------------------------------------------------

# Sorted list of available ONNX standard opset versions in spox.
_SPOX_MAIN_VERSIONS = (17, 18, 19, 20, 21, 22, 23, 24)
# Latest available ai.onnx.ml version in spox.
_SPOX_ML_VERSION = 3


def _resolve_main_version(requested: int) -> int:
    """Returns the highest spox opset version that does not exceed *requested*."""
    best = _SPOX_MAIN_VERSIONS[0]
    for v in _SPOX_MAIN_VERSIONS:
        if v <= requested:
            best = v
        else:
            break
    return best


def _get_op_module(domain: str, version: int):
    """
    Returns the spox opset module for *domain* and *version*.

    Supported domains:

    * ``""`` (ONNX standard) — maps to ``spox.opset.ai.onnx.vN``
    * ``"ai.onnx.ml"`` — maps to ``spox.opset.ai.onnx.ml.v3``

    :raises ImportError: When spox is not installed.
    :raises NotImplementedError: When the domain is not supported.
    """
    if domain == "":
        resolved = _resolve_main_version(version)
        return importlib.import_module(f"spox.opset.ai.onnx.v{resolved}")
    if domain == "ai.onnx.ml":
        return importlib.import_module(f"spox.opset.ai.onnx.ml.v{_SPOX_ML_VERSION}")
    raise NotImplementedError(
        f"Domain {domain!r} is not directly supported by SpoxGraphBuilder. "
        "Use make_node() with an ONNX proto fallback for custom domains."
    )


# ---------------------------------------------------------------------------
# Helper: convert an ONNX elem-type integer to a numpy dtype accepted by spox
# ---------------------------------------------------------------------------


def _onnx_elem_type_to_np_dtype(elem_type: int) -> np.dtype:
    """Convert an ONNX ``TensorProto`` element type int to a :class:`numpy.dtype`."""
    return np.dtype(tensor_dtype_to_np_dtype(elem_type))


def _np_dtype_to_onnx_elem_type(dtype) -> int:
    """Convert a numpy dtype to an ONNX ``TensorProto`` element type int."""
    from spox._type_system import dtype_to_tensor_type

    return dtype_to_tensor_type(np.dtype(dtype))


# ---------------------------------------------------------------------------
# Helpers: introspect spox constructor signatures
# ---------------------------------------------------------------------------


def _positional_params(ctor: Callable) -> List[inspect.Parameter]:
    """Returns the positional (non-keyword-only) parameters of *ctor*."""
    sig = inspect.signature(ctor)
    return [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
    ]


def _keyword_only_params(ctor: Callable) -> Dict[str, inspect.Parameter]:
    """Returns a dict of keyword-only parameters of *ctor*."""
    sig = inspect.signature(ctor)
    return {p.name: p for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY}


def _is_sequence_var_param(param: inspect.Parameter, ctor: Callable) -> bool:
    """Returns ``True`` when *param* expects a ``Sequence[Var]`` (variadic inputs)."""
    try:
        hints = typing.get_type_hints(ctor)
    except Exception:
        return False
    hint = hints.get(param.name)
    if hint is None:
        return False
    origin = typing.get_origin(hint)
    if origin is None:
        return False
    return issubclass(origin, collections.abc.Sequence)


def _is_dtype_like_param(param: inspect.Parameter, ctor: Callable) -> bool:
    """Returns ``True`` when *param* expects a dtype-like value (e.g. for ``Cast.to``)."""
    try:
        hints = typing.get_type_hints(ctor)
    except Exception:
        return False
    hint = hints.get(param.name)
    if hint is None:
        return False
    return hint is npt.DTypeLike


# ---------------------------------------------------------------------------
# Opset helper (implements OpsetProtocol)
# ---------------------------------------------------------------------------


class SpoxGraphBuilderOpset(OpsetProtocol):
    """Implements :class:`yobx.typing.OpsetProtocol` for :class:`SpoxGraphBuilder`."""

    def __init__(self, builder: SpoxGraphBuilder) -> None:
        self._builder = builder

    def __getattr__(self, op_type: str) -> Callable[..., Union[str, Tuple[str, ...]]]:
        return partial(self._builder._make_node, op_type)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


class SpoxGraphBuilder(GraphBuilderExtendedProtocol):
    """
    Bridge builder that exposes a yobx-compatible API over :epkg:`spox`.

    :param target_opset_or_opsets: Either a single opset version (``int``) or
        a mapping ``{domain: version}`` (``Dict[str, int]``).  For example
        ``18`` or ``{"": 18, "ai.onnx.ml": 3}``.
    :param ir_version: unused; kept for API parity with other builders.

    The builder maps string tensor names to :class:`spox.Var` objects and
    delegates graph construction to :epkg:`spox`'s opset modules.  Calling
    :meth:`to_onnx` runs ``spox.build`` and returns an
    :class:`onnx.ModelProto`.

    Typical usage::

        from yobx.builder.spox import SpoxGraphBuilder
        from onnx import TensorProto

        g = SpoxGraphBuilder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        w = g.make_initializer("W", np.eye(4, dtype=np.float32))
        (y,) = g.make_node("MatMul", ["X", w], 1)
        g.make_tensor_output(y)
        proto = g.to_onnx()
    """

    def __init__(
        self,
        target_opset_or_opsets: Union[int, Dict[str, int]],
        ir_version: Optional[int] = None,
    ) -> None:
        if isinstance(target_opset_or_opsets, int):
            self.opsets: Dict[str, int] = {"": target_opset_or_opsets}
        elif isinstance(target_opset_or_opsets, dict):
            self.opsets = dict(target_opset_or_opsets)
        else:
            raise NotImplementedError(
                f"Type {type(target_opset_or_opsets)} is not supported for "
                "target_opset_or_opsets."
            )

        # Mapping: user-visible string name → spox.Var
        self._name_to_var: Dict[str, Any] = {}
        # Ordered collections for graph inputs/outputs
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        # Side-channel type/shape overrides (for set_type / set_shape)
        self._type_map: Dict[str, int] = {}
        self._shape_map: Dict[str, DYNAMIC_SHAPE] = {}
        # Counter for auto-generated unique names
        self._unique_names: Set[str] = set()
        self._op = SpoxGraphBuilderOpset(self)

    # ------------------------------------------------------------------
    # GraphBuilderExtendedProtocol: core properties / methods
    # ------------------------------------------------------------------

    @property
    def main_opset(self) -> int:
        """Returns the opset version for the main (``""``/ONNX) domain."""
        return self.opsets[""]

    @property
    def op(self) -> OpsetProtocol:
        """Returns the opset helper (``g.op.Relu(x)``)."""
        return self._op

    def unique_name(self, prefix: str) -> str:
        """Returns a unique name derived from *prefix*."""
        if prefix not in self._unique_names:
            self._unique_names.add(prefix)
            return prefix
        i = 2
        candidate = f"{prefix}{i}"
        while candidate in self._unique_names:
            i += 1
            candidate = f"{prefix}{i}"
        self._unique_names.add(candidate)
        return candidate

    def set_type_shape_unary_op(
        self,
        name: str,
        input_name: str,
        itype: Optional[int] = None,
    ) -> bool:
        return set_type_shape_unary_op(self, name, input_name, itype)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # GraphBuilderProtocol: opset management
    # ------------------------------------------------------------------

    def get_opset(self, domain: str, exc: bool = True) -> Optional[int]:
        """Returns the opset version for *domain*."""
        if exc:
            assert (
                domain in self.opsets
            ), f"Domain {domain!r} is not registered. opsets={self.opsets!r}."
        return self.opsets.get(domain, None)

    def set_opset(self, domain: str, version: int = 1) -> None:
        """Registers *version* for *domain*, or no-ops if already set to the same value."""
        if domain in self.opsets:
            assert version == self.opsets[domain], (
                f"Version mismatch for domain={domain!r}: "
                f"existing={self.opsets[domain]}, new={version}."
            )
            return
        self.opsets[domain] = version

    def add_domain(self, domain: str, version: int = 1) -> None:
        """Deprecated alias for :meth:`set_opset`."""
        self.set_opset(domain, version)

    def has_opset(self, domain: str) -> int:
        """Returns the opset version for *domain*, or ``0`` if not registered."""
        return self.opsets.get(domain, 0)

    # ------------------------------------------------------------------
    # GraphBuilderProtocol: name / type / shape helpers
    # ------------------------------------------------------------------

    def has_name(self, name: str) -> bool:
        """Returns ``True`` when *name* is known in this graph."""
        return name in self._name_to_var

    def has_type(self, name: str) -> bool:
        """Returns ``True`` when *name* has a known element type."""
        if name in self._type_map:
            return True
        var = self._name_to_var.get(name)
        if var is None:
            return False
        t = var.type
        return t is not None and hasattr(t, "dtype") and t.dtype is not None

    def get_type(self, name: str) -> int:
        """Returns the ONNX element type integer for *name*, or ``0`` if unknown."""
        if name in self._type_map:
            return self._type_map[name]
        var = self._name_to_var.get(name)
        if var is None:
            return 0
        t = var.type
        if t is None or not hasattr(t, "dtype") or t.dtype is None:
            return 0
        try:
            return _np_dtype_to_onnx_elem_type(t.dtype)
        except Exception:
            return 0

    def set_type(self, name: str, itype: int) -> None:
        """Overrides the element type for *name* in the side-channel type map."""
        self._type_map[name] = itype

    def has_shape(self, name: str) -> bool:
        """Returns ``True`` when *name* has a known shape."""
        if name in self._shape_map:
            return True
        var = self._name_to_var.get(name)
        if var is None:
            return False
        t = var.type
        if t is None or not hasattr(t, "shape"):
            return False
        return t.shape is not None

    def get_shape(self, name: str) -> DYNAMIC_SHAPE:
        """Returns the shape for *name* as a tuple, or raises ``AssertionError``."""
        if name in self._shape_map:
            return self._shape_map[name]
        assert name in self._name_to_var, f"Name {name!r} is not registered."
        var = self._name_to_var[name]
        t = var.type
        assert t is not None and hasattr(t, "shape"), f"Name {name!r} has no shape."
        shape = t.shape
        assert shape is not None, f"Name {name!r} has a shape but it is None."
        return tuple(d if isinstance(d, (int, str)) else None for d in shape)

    def set_shape(self, name: str, shape: DYNAMIC_SHAPE, allow_zero: bool = False) -> None:
        """Stores *shape* in the side-channel shape map for *name*."""
        assert shape is not None, f"shape cannot be None for name={name!r}"
        assert (
            allow_zero or not shape or 0 not in shape
        ), f"Shape {shape} for name={name!r} contains zero."
        self._shape_map[name] = shape

    def has_device(self, name: str) -> bool:
        """Always returns ``False`` — device tracking is not supported."""
        return False

    def get_device(self, name: str) -> int:
        raise NotImplementedError(
            f"Device for {name!r} is not available with {self.__class__.__name__}."
        )

    def onnx_dtype_to_np_dtype(self, itype: int) -> np.dtype:
        """See :func:`yobx.helpers.onnx_helper.tensor_dtype_to_np_dtype`."""
        return np.dtype(tensor_dtype_to_np_dtype(itype))

    def get_debug_msg(self) -> str:
        """Returns a short debug message listing known names."""
        return f" known_names={sorted(self._name_to_var)}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def input_names(self) -> List[str]:
        """Returns the list of graph input names."""
        return list(self._input_names)

    @property
    def output_names(self) -> List[str]:
        """Returns the list of graph output names."""
        return list(self._output_names)

    def _get_var(self, name: str) -> Any:
        """Returns the :class:`spox.Var` for *name*."""
        try:
            return self._name_to_var[name]
        except KeyError:
            raise KeyError(
                f"Name {name!r} is not known. Known: {sorted(self._name_to_var)}"
            ) from None

    def _register(self, name: str, var: Any) -> None:
        """Registers *var* under *name* in the internal name registry."""
        self._name_to_var[name] = var
        self._unique_names.add(name)

    def _arg_to_var(self, arg: Any) -> Any:
        """
        Converts a single op argument to a :class:`spox.Var`.

        * ``str`` — looks up the var by name.
        * ``""`` — optional absent input → ``None``.
        * :class:`numpy.ndarray` / scalar — creates an inline ``Constant`` node.
        * ``None`` — optional absent input → ``None``.
        """
        if arg is None:
            return None
        if isinstance(arg, str):
            if arg == "":
                return None
            return self._get_var(arg)
        # numpy array or scalar: auto-create a Constant
        if isinstance(arg, (int, float)):
            if isinstance(arg, int):
                arr = np.array(arg, dtype=np.int64)
            else:
                arr = np.array(arg, dtype=np.float32)
        elif isinstance(arg, np.ndarray):
            arr = arg
        elif hasattr(arg, "dtype") and hasattr(arg, "shape"):
            arr = np.array(arg)
        else:
            raise TypeError(
                f"Cannot convert argument of type {type(arg)} to a spox.Var. "
                "Supported: str (registered name), numpy.ndarray, int, float."
            )
        init_name = self.make_initializer("", arr)
        return self._get_var(init_name)

    def _adapt_kwarg(self, name: str, value: Any, ctor: Callable) -> Any:
        """
        Converts *value* for kwarg *name* of *ctor* to the type spox expects.

        Currently handles ``DTypeLike`` parameters: when the caller passes an
        ONNX element-type integer (e.g. ``onnx.TensorProto.INT64 == 7``), the
        value is converted to a :class:`numpy.dtype`.
        """
        kw_params = _keyword_only_params(ctor)
        param = kw_params.get(name)
        if param is None:
            return value
        if isinstance(value, int) and _is_dtype_like_param(param, ctor):
            return _onnx_elem_type_to_np_dtype(value)
        return value

    def _dispatch_op(
        self,
        op_type: str,
        input_vars: List[Any],
        output_names: Optional[List[str]],
        domain: str,
        **kwargs: Any,
    ) -> Tuple[str, ...]:
        """
        Calls the spox constructor for (*domain*, *op_type*) with *input_vars*
        and *kwargs*, registers the results under *output_names* (or auto-names),
        and returns the final names as a tuple.
        """
        version = self.opsets.get(domain, self.main_opset) if domain else self.main_opset
        op_module = _get_op_module(domain, version)

        constructors = getattr(op_module, "_CONSTRUCTORS", {})
        ctor = constructors.get(op_type)
        if ctor is None:
            raise NotImplementedError(
                f"Operator {op_type!r} is not available in domain {domain!r} "
                f"for opset {version}. "
                f"Available: {sorted(constructors)}"
            )

        # Adapt keyword arguments (e.g. integer dtype → numpy dtype for Cast)
        adapted_kwargs = {k: self._adapt_kwarg(k, v, ctor) for k, v in kwargs.items()}

        # Determine how to pass positional inputs
        pos_params = _positional_params(ctor)
        if pos_params and _is_sequence_var_param(pos_params[0], ctor):
            # The first positional param expects a Sequence[Var] (e.g. Concat, Sum).
            # Pass all inputs as a single list.
            result = ctor(input_vars, **adapted_kwargs)
        else:
            # Each positional param receives one Var.
            result = ctor(*input_vars, **adapted_kwargs)

        # Normalise the result to a list of Vars
        if isinstance(result, spox.Var):
            result_list: List[Any] = [result]
        else:
            result_list = list(result)

        # Register each output under the requested (or auto-generated) name
        final_names: List[str] = []
        for i, var in enumerate(result_list):
            if output_names and i < len(output_names):
                n = output_names[i]
            else:
                n = self.unique_name(op_type)
            self._register(n, var)
            final_names.append(n)

        return tuple(final_names)

    # ------------------------------------------------------------------
    # Core builder API
    # ------------------------------------------------------------------

    def make_tensor_input(
        self,
        name: str,
        elem_type: Optional[int] = None,
        shape: Optional[Sequence[Optional[Union[int, str]]]] = None,
        device: Optional[int] = None,
    ) -> str:
        """
        Adds a graph input and returns its name.

        :param name: Input tensor name.
        :param elem_type: ONNX element type (e.g. ``TensorProto.FLOAT``).
        :param shape: Tensor shape.
        :param device: unused.
        :return: The registered name (same as *name*).
        """
        if elem_type:
            np_dtype = _onnx_elem_type_to_np_dtype(elem_type)
            spox_shape = tuple(shape) if shape is not None else None
            spox_type = spox.Tensor(np_dtype, spox_shape)
        else:
            # Unknown type — create an untyped argument; spox may warn.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spox_type = spox.Tensor(np.float32)  # fallback

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var = spox.argument(spox_type)

        self._register(name, var)
        self._input_names.append(name)
        if elem_type:
            self._type_map[name] = elem_type
        if shape is not None:
            self._shape_map[name] = tuple(shape)
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
        Registers an existing value as a graph output and returns its name.

        :param name: Name (or list of names) of the tensor(s) to mark as output(s).
        :param elem_type: Optional element type hint.
        :param shape: Optional shape hint.
        :param indexed: unused.
        :param allow_untyped_output: allow output with no type/shape.
        :return: The name (or list of names).
        """
        if isinstance(name, list):
            return [
                self.make_tensor_output(n, elem_type=elem_type, shape=shape)  # type: ignore[misc]
                for n in name
            ]

        assert (
            name in self._name_to_var
        ), f"Cannot mark {name!r} as output — it is not registered."
        self._output_names.append(name)
        if elem_type is not None and name not in self._type_map:
            self._type_map[name] = elem_type
        if shape is not None and name not in self._shape_map:
            self._shape_map[name] = tuple(shape)
        return name

    def make_initializer(self, name: str, value: Any) -> str:
        """
        Adds an initializer (Constant node) and returns its name.

        :param name: Name for the initializer.  May be ``""`` for auto-naming.
        :param value: Initializer data: :class:`numpy.ndarray`, ``int``, ``float``,
            or :class:`onnx.TensorProto`.
        :return: The registered name.
        """
        if not name:
            name = self.unique_name("init_")

        # Convert to numpy array
        if isinstance(value, onnx.TensorProto):
            arr = onnx.numpy_helper.to_array(value)
        elif isinstance(value, int):
            arr = np.array(value, dtype=np.int64)
        elif isinstance(value, float):
            arr = np.array(value, dtype=np.float32)
        elif isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "dtype") and hasattr(value, "shape"):
            arr = np.array(value)
        else:
            raise TypeError(
                f"Cannot convert initializer {name!r} of type {type(value)} to a numpy array."
            )

        # Create a constant Var using the appropriate opset module
        op_module = _get_op_module("", self.main_opset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var = op_module.constant(value=arr)

        self._register(name, var)
        self._type_map[name] = _np_dtype_to_onnx_elem_type(arr.dtype)
        self._shape_map[name] = tuple(arr.shape)
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
        Creates an ONNX node and returns its output name(s).

        :param op_type: ONNX operator type.
        :param inputs: Input tensor name(s).
        :param outputs: Number of outputs (``int``), a single name (``str``),
            or a list of names.
        :param domain: Operator domain.
        :param attributes: Additional :class:`onnx.AttributeProto` instances.
        :param name: Optional node name (ignored by spox).
        :param kwargs: Operator attributes.
        :return: Single output name or tuple of names.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        # Convert AttributeProto → kwargs entries
        if attributes:
            for attr in attributes:
                if attr.name not in kwargs:
                    kwargs[attr.name] = self._attr_proto_to_python(attr)

        # Resolve output names list
        if isinstance(outputs, int):
            output_names: Optional[List[str]] = None
            output_count = outputs
        elif isinstance(outputs, str):
            output_names = [outputs]
            output_count = 1
        else:
            output_names = list(outputs)
            output_count = len(output_names)

        # Convert input strings → Var objects
        input_vars = [self._arg_to_var(inp) for inp in inputs]

        result_names = self._dispatch_op(
            op_type,
            input_vars,
            output_names,
            domain,
            **kwargs,
        )

        # When fewer names were requested than produced, trim (rare edge case)
        if output_names is None:
            result_names = result_names[:output_count]

        if len(result_names) == 1:
            return result_names[0]
        return result_names

    def _make_node(
        self,
        op_type: str,
        *args: Any,
        domain: str = "",
        outputs: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, ...]]:
        """
        Internal handler called via ``g.op.OpName(...)`` dispatch.

        Positional *args* are the input tensors (strings or numpy arrays).
        ``domain``, ``outputs``, and ``name`` (popped from *kwargs*) are
        metadata; all other keyword arguments are operator attributes.
        """
        # Strip metadata kwargs that are not op attributes
        kwargs.pop("name", None)

        # Convert positional args → Var objects
        input_vars = [self._arg_to_var(a) for a in args]

        output_names_list: Optional[List[str]] = list(outputs) if outputs else None

        result_names = self._dispatch_op(
            op_type,
            input_vars,
            output_names_list,
            domain,
            **kwargs,
        )

        if len(result_names) == 1:
            return result_names[0]
        return result_names

    @staticmethod
    def _attr_proto_to_python(attr: onnx.AttributeProto) -> Any:
        """Converts an :class:`onnx.AttributeProto` to a plain Python value."""
        return attr_proto_to_python(attr)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_onnx(self) -> onnx.ModelProto:
        """Exports the accumulated graph as an :class:`onnx.ModelProto`."""
        inputs_dict = {n: self._name_to_var[n] for n in self._input_names}
        outputs_dict = {n: self._name_to_var[n] for n in self._output_names}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proto = spox.build(inputs_dict, outputs_dict)

        return proto
