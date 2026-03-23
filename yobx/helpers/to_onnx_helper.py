from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from onnx import ValueInfoProto
from ..typing import GraphBuilderExtendedProtocol
from .onnx_helper import np_dtype_to_tensor_dtype


def extract_value_info_proto(vip: ValueInfoProto) -> Tuple[str, int, Optional[Tuple]]:
    """Extract ``(name, elem_type, shape)`` from a :class:`onnx.ValueInfoProto`.

    :param vip: an ONNX value-info descriptor
    :return: tuple ``(name, elem_type, shape)`` where *shape* is ``None`` when
        no shape information is present, and otherwise a tuple whose elements
        are ``int`` for static dimensions, a non-empty ``str`` for symbolic
        dimensions, and an auto-generated name such as ``"_unk_0_"`` for
        dimensions with no value and no symbolic name.
    """
    name = vip.name
    tt = vip.type.tensor_type
    elem_type = tt.elem_type
    if tt.HasField("shape"):
        shape_dims = [
            (dim.dim_param if dim.dim_param else (dim.dim_value or f"dim_{name}_{idim}"))
            for idim, dim in enumerate(tt.shape.dim)
        ]
        shape: Optional[Tuple] = tuple(shape_dims)
    else:
        shape = None
    return name, elem_type, shape


def is_arg_tuple_spec(arg: Any) -> bool:
    """Return True if *arg* is a ``(name, dtype, shape)`` input specification.

    Supported format::

        ('input_name', np.float32, ('N', 4))

    where

    * ``name`` is a :class:`str`
    * ``dtype`` is a :class:`numpy.dtype` instance, a numpy scalar-type class
      (e.g. ``np.float32``), or anything accepted by ``np.dtype(...)``
    * ``shape`` is a :class:`tuple` or :class:`list` whose elements are
      integers (static dimensions) or strings (symbolic dimensions).
    """
    if not (isinstance(arg, tuple) and len(arg) == 3 and isinstance(arg[0], str)):
        return False
    if not isinstance(arg[2], (tuple, list)):
        return False
    if not all(isinstance(d, (int, str)) for d in arg[2]):
        return False
    return True


def register_inputs(
    g: GraphBuilderExtendedProtocol,
    args: Tuple[Any, ...],
    input_names: Optional[Sequence[str]],
    dynamic_shapes: Optional[Tuple[Dict[int, str]]],
) -> List[str]:
    """Resolve *input_names* and register each input with the graph builder *g*.

    :param g: graph builder (:class:`~yobx.typing.GraphBuilderExtendedProtocol`)
    :param args: input descriptors — numpy arrays,
        :class:`onnx.ValueInfoProto` objects, or ``(name, dtype, shape)`` tuples
    :param input_names: optional explicit names; overrides names embedded in
        :class:`~onnx.ValueInfoProto` / tuple descriptors when provided
    :param dynamic_shapes: per-input axis-to-symbol mapping used only for
        numpy-array inputs; ``None`` defaults to ``{0: "batch"}``
    :return: the resolved list of input names (same length as *args*)
    """
    if input_names:
        if len(input_names) != len(args):
            raise ValueError(f"Length mismatch: {len(args)=} but input_names={input_names!r}")
        resolved_names: List[str] = list(input_names)
    else:
        # Derive default input names; for ValueInfoProto or tuple specs use the
        # embedded name.
        default_names = []
        for j, arg in enumerate(args):
            if isinstance(arg, ValueInfoProto):
                default_names.append(arg.name or (f"X{j}" if len(args) > 1 else "X"))
            elif is_arg_tuple_spec(arg):
                default_names.append(arg[0])
            else:
                default_names.append("X" if len(args) == 1 else f"X{j}")
        resolved_names = default_names

    for i, (name, arg) in enumerate(zip(resolved_names, args)):
        if isinstance(arg, ValueInfoProto):
            # Use name/type/shape directly from the ValueInfoProto.
            _, elem_type, shape = extract_value_info_proto(arg)
            g.make_tensor_input(name, elem_type, shape, device=-1)
        elif is_arg_tuple_spec(arg):
            # Use name/dtype/shape directly from the (name, dtype, shape) tuple.
            _, arg_dtype, arg_shape = arg
            elem_type = np_dtype_to_tensor_dtype(np.dtype(arg_dtype))
            g.make_tensor_input(name, elem_type, tuple(arg_shape), device=-1)
        else:
            if dynamic_shapes:
                ds = dynamic_shapes[i]
            else:
                ds = {0: "batch"}
            shape = list(arg.shape)  # type: ignore
            for axis, dim in ds.items():
                shape[axis] = dim  # type: ignore
            g.make_tensor_input(  # type: ignore
                name, np_dtype_to_tensor_dtype(arg.dtype), tuple(shape), device=-1  # type: ignore
            )  # type: ignore

    return resolved_names
