"""
Converters for TF shape-manipulation ops:
``Reshape``, ``Squeeze``, ``Shape``, ``StridedSlice``, ``Pack``,
``ExpandDims``, ``Transpose``, ``Cast``.

Reshape / squeeze
-----------------
``Reshape``, ``Squeeze``

Type casting
------------
``Cast``

Shape / indexing
----------------
``Shape``, ``StridedSlice``

Stack / pack
------------
``Pack``

Dimension insertion / permutation
----------------------------------
``ExpandDims``, ``Transpose``
"""

from typing import Any, Dict, List

import numpy as np
from onnx import TensorProto
import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol

# Mapping from TF dtype to ONNX TensorProto type.
_TF_DTYPE_TO_ONNX = {
    tf.float16.as_datatype_enum: TensorProto.FLOAT16,
    tf.float32.as_datatype_enum: TensorProto.FLOAT,
    tf.float64.as_datatype_enum: TensorProto.DOUBLE,
    tf.bfloat16.as_datatype_enum: TensorProto.BFLOAT16,
    tf.int8.as_datatype_enum: TensorProto.INT8,
    tf.int16.as_datatype_enum: TensorProto.INT16,
    tf.int32.as_datatype_enum: TensorProto.INT32,
    tf.int64.as_datatype_enum: TensorProto.INT64,
    tf.uint8.as_datatype_enum: TensorProto.UINT8,
    tf.uint16.as_datatype_enum: TensorProto.UINT16,
    tf.uint32.as_datatype_enum: TensorProto.UINT32,
    tf.uint64.as_datatype_enum: TensorProto.UINT64,
    tf.bool.as_datatype_enum: TensorProto.BOOL,
    tf.string.as_datatype_enum: TensorProto.STRING,
}


@register_tf_op_converter("Reshape")
def convert_reshape(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``Reshape`` → ONNX ``Reshape``.

    The TF ``shape`` input (second input) may be int32; it is cast to int64
    as required by the ONNX ``Reshape`` specification.

    When ONNX shape inference cannot determine the output shape (e.g. because
    the shape tensor contains a dynamic batch dimension), the TF-inferred output
    shape is used as a fallback so that downstream shape-inference assertions are
    satisfied.
    """
    shape_i64 = g.op.Cast(op.inputs[1].name, to=TensorProto.INT64, name=f"{op.name}_shape_cast")
    result = g.op.Reshape(op.inputs[0].name, shape_i64, outputs=outputs[:1], name=op.name)

    # Fall back to TF's known output shape when ONNX inference did not produce one.
    if not g.has_shape(outputs[0]):
        tf_dims = op.outputs[0].shape.as_list()
        if tf_dims is not None and g.has_shape(op.inputs[0].name):
            in_shape = g.get_shape(op.inputs[0].name)
            # Build a map from position-in-output to a symbolic or concrete dimension value.
            # For None (dynamic) dims, reuse the symbolic name from the input if available.
            in_dynamic = [d for d in in_shape if isinstance(d, str)]

            def _map_dim(d: Any, idx: int) -> Any:
                if d is not None:
                    return d
                # Use input's symbolic dim at the same position when present.
                if idx < len(in_shape) and isinstance(in_shape[idx], str):
                    return in_shape[idx]
                # Otherwise fall back to the first dynamic input dim or a generic name.
                return in_dynamic[0] if in_dynamic else f"rd{idx}"

            onnx_shape = tuple(_map_dim(d, i) for i, d in enumerate(tf_dims))
            g.set_shape(outputs[0], onnx_shape)
            if g.has_type(op.inputs[0].name):
                g.set_type(outputs[0], g.get_type(op.inputs[0].name))

    return result


@register_tf_op_converter("Squeeze")
def convert_squeeze(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``Squeeze`` → ONNX ``Squeeze``.

    When ``squeeze_dims`` is empty all size-1 dimensions are removed (matching
    TF's default behaviour).  When specific axes are given they are passed as
    an int64 tensor input to the ONNX node (opset ≥ 13 API).
    """
    squeeze_dims = list(op.get_attr("squeeze_dims"))
    if squeeze_dims:
        axes = np.array(squeeze_dims, dtype=np.int64)
        return g.op.Squeeze(op.inputs[0].name, axes, outputs=outputs[:1], name=op.name)
    return g.op.Squeeze(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Cast")
def convert_cast(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``Cast`` → ONNX ``Cast``.

    The target dtype is read from the TF op's ``DstT`` attribute and
    mapped to the corresponding ONNX ``TensorProto`` integer type.
    """
    dst_tf = op.get_attr("DstT").as_datatype_enum
    dst_onnx = _TF_DTYPE_TO_ONNX.get(dst_tf)
    if dst_onnx is None:
        raise NotImplementedError(
            f"TF dtype {op.get_attr('DstT')} has no ONNX TensorProto mapping."
        )
    return g.op.Cast(op.inputs[0].name, to=dst_onnx, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Shape")
def convert_shape(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``Shape`` → ONNX ``Shape``.

    TF's ``Shape`` op may output ``int32`` (via the ``out_type`` attribute),
    but the ONNX ``Shape`` operator always returns ``int64``.  The output is
    kept as ``int64`` throughout the ONNX graph; downstream converters
    (e.g. ``StridedSlice``, ``Pack``) cast their non-int64 constant inputs
    to match.
    """
    return g.op.Shape(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("StridedSlice")
def convert_strided_slice(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``StridedSlice`` → ONNX ``Gather`` or ``Slice`` (+ optional
    ``Squeeze``).

    Only the common cases are supported:

    * No ``begin_mask``, ``end_mask``, ``ellipsis_mask``, or
      ``new_axis_mask`` bits set.
    * ``shrink_axis_mask`` is supported by using ``Gather`` with a scalar
      index for the typical single-element-extraction pattern (e.g. as
      emitted by ``tf.keras.layers.Flatten``).

    This covers the pattern generated by ``tf.keras.layers.Flatten`` and
    similar common usages of ``StridedSlice`` on shape tensors.
    """
    begin_mask = op.get_attr("begin_mask")
    end_mask = op.get_attr("end_mask")
    ellipsis_mask = op.get_attr("ellipsis_mask")
    new_axis_mask = op.get_attr("new_axis_mask")
    shrink_axis_mask = op.get_attr("shrink_axis_mask")

    if begin_mask or end_mask or ellipsis_mask or new_axis_mask:
        raise NotImplementedError(
            f"StridedSlice with non-zero begin_mask={begin_mask}, "
            f"end_mask={end_mask}, ellipsis_mask={ellipsis_mask}, "
            f"new_axis_mask={new_axis_mask} is not supported."
        )

    x = op.inputs[0].name

    if shrink_axis_mask:
        # Common pattern: extract a single element and drop the dimension.
        # Use Gather with a scalar index for reliable shape inference.
        try:
            begin_val = tf.make_ndarray(op.inputs[1].op.get_attr("value"))
        except (AttributeError, ValueError):
            begin_val = None

        if begin_val is not None:
            # Determine which axes are shrunk (bits set in shrink_axis_mask).
            shrink_axes = [i for i in range(32) if (shrink_axis_mask >> i) & 1]
            if len(shrink_axes) == 1:
                axis = shrink_axes[0]
                # Extract the element at begin[axis] along the given axis.
                idx = (
                    int(begin_val[axis]) if hasattr(begin_val, "__getitem__") else int(begin_val)
                )
                idx_scalar = np.array(idx, dtype=np.int64)
                return g.op.Gather(x, idx_scalar, axis=axis, outputs=outputs[:1], name=op.name)

    # General case: cast begin/end to int64 and use Slice.
    begin_i64 = g.op.Cast(op.inputs[1].name, to=TensorProto.INT64, name=f"{op.name}_begin_cast")
    end_i64 = g.op.Cast(op.inputs[2].name, to=TensorProto.INT64, name=f"{op.name}_end_cast")
    sliced = g.op.Slice(x, begin_i64, end_i64, name=f"{op.name}_slice")

    if shrink_axis_mask:
        axes_to_squeeze = [i for i in range(32) if (shrink_axis_mask >> i) & 1]
        axes_arr = np.array(axes_to_squeeze, dtype=np.int64)
        return g.op.Squeeze(sliced, axes_arr, outputs=outputs[:1], name=op.name)

    return g.op.Identity(sliced, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Pack")
def convert_pack(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``Pack`` (``tf.stack``) → ONNX ``Unsqueeze`` + ``Concat``.

    ``Pack`` stacks N tensors along a new axis.  In ONNX this is expressed
    as unsqueezing each input along the new axis and then concatenating them.

    Each input is cast to ``int64`` so that the resulting tensor can be used
    as an ONNX ``Reshape`` shape argument (which requires ``int64``).
    """
    axis = int(op.get_attr("axis"))
    inputs = [inp.name for inp in op.inputs]
    axes_arr = np.array([axis], dtype=np.int64)

    unsqueezed = []
    for i, inp in enumerate(inputs):
        # Cast to int64: shape computations in ONNX use int64.
        casted = g.op.Cast(inp, to=TensorProto.INT64, name=f"{op.name}_cast_{i}")
        us = g.op.Unsqueeze(casted, axes_arr, name=f"{op.name}_us_{i}")
        unsqueezed.append(us)

    if len(unsqueezed) == 1:
        return g.op.Identity(unsqueezed[0], outputs=outputs[:1], name=op.name)
    return g.op.Concat(*unsqueezed, axis=axis, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("ExpandDims")
def convert_expand_dims(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``ExpandDims`` (``tf.expand_dims``) → ONNX ``Unsqueeze``.

    The ``dim`` input (second input) must be a scalar constant; its value is
    read and forwarded as the ``axes`` argument of ``Unsqueeze``.
    """
    dim_op = op.inputs[1].op
    dim_val = int(tf.make_ndarray(dim_op.get_attr("value")))
    axes_arr = np.array([dim_val], dtype=np.int64)
    return g.op.Unsqueeze(op.inputs[0].name, axes_arr, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Transpose")
def convert_transpose(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``Transpose`` → ONNX ``Transpose``.

    The permutation is given as a second input tensor (a 1-D int32/int64
    constant); its value is read and forwarded as the ``perm`` attribute.
    """
    perm_op = op.inputs[1].op
    perm = list(tf.make_ndarray(perm_op.get_attr("value")).astype(int))
    return g.op.Transpose(op.inputs[0].name, perm=perm, outputs=outputs[:1], name=op.name)
