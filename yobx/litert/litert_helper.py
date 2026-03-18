"""
Minimal pure-Python :epkg:`TFLite`/:epkg:`LiteRT` FlatBuffer parser and
helper utilities for the LiteRT→ONNX converter.

The parser requires no external dependencies beyond the Python standard
library. It reads the binary FlatBuffer format that every ``.tflite``
file uses and exposes the model graph as plain Python dataclasses so
that the converter can inspect operator types, tensor shapes, weights,
and graph connectivity.

TFLite dtype enum mapping
-------------------------

=====  ===============================
Value  TFLite ``TensorType``
=====  ===============================
0      FLOAT32
1      FLOAT16
2      INT32
3      UINT8
4      INT64
5      STRING
6      BOOL
7      INT16
8      COMPLEX64
9      INT8
10     FLOAT64
11     COMPLEX128
12     UINT64
14     UINT32
15     UINT16
16     INT4
=====  ===============================
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# TFLite tensor type → numpy dtype
# ---------------------------------------------------------------------------

_TFLITE_DTYPE_MAP: Dict[int, np.dtype] = {
    0: np.dtype("float32"),
    1: np.dtype("float16"),
    2: np.dtype("int32"),
    3: np.dtype("uint8"),
    4: np.dtype("int64"),
    6: np.dtype("bool"),
    7: np.dtype("int16"),
    9: np.dtype("int8"),
    10: np.dtype("float64"),
    12: np.dtype("uint64"),
    14: np.dtype("uint32"),
    15: np.dtype("uint16"),
}


def litert_dtype_to_np_dtype(dtype_int: int) -> np.dtype:
    """Map a TFLite ``TensorType`` integer to a :class:`numpy.dtype`.

    :param dtype_int: TFLite dtype integer from the FlatBuffer
    :return: corresponding :class:`numpy.dtype`
    :raises ValueError: if the dtype is not supported
    """
    dt = _TFLITE_DTYPE_MAP.get(dtype_int)
    if dt is None:
        raise ValueError(
            f"Unsupported TFLite TensorType={dtype_int}. "
            f"Supported types: {sorted(_TFLITE_DTYPE_MAP)}"
        )
    return dt


# ---------------------------------------------------------------------------
# Minimal FlatBuffer reader (pure Python, no external deps)
# ---------------------------------------------------------------------------

# TFLite file identifiers used across schema versions.
_TFLITE_IDENTIFIER = b"TFL3"
# Four-space placeholder used by some early / custom serialisation tools that
# write a blank file identifier rather than the version-specific string.
_TFLITE_BLANK_IDENTIFIER = b"    "


class _FlatBuf:
    """Minimal read-only FlatBuffer reader for TFLite models.

    Implements just enough of the FlatBuffer wire format to extract the
    structures needed by the converter: tables, vectors, scalars, and
    strings.  Unions are handled by reading the type byte then delegating
    to the caller.

    FlatBuffer layout summary
    -------------------------
    * Every table starts with a 4-byte *signed* offset (``soffset_t``) that
      points **back** to its vtable: ``vtable_pos = table_pos - soffset``.
    * A vtable begins with two ``uint16`` fields (vtable byte-size and
      object byte-size) followed by one ``uint16`` per object field.  An
      entry of ``0`` means the field is absent (use the declared default).
    * Offsets to nested tables, vectors, and strings are *relative*:
      the absolute position is ``field_start + uint32(field_start)``.
    """

    __slots__ = ("buf",)

    def __init__(self, buf: bytes) -> None:
        self.buf: bytes = buf if isinstance(buf, bytes) else bytes(buf)

    # ------------------------------------------------------------------ #
    # Low-level scalar readers                                             #
    # ------------------------------------------------------------------ #

    def _u8(self, pos: int) -> int:
        return struct.unpack_from("<B", self.buf, pos)[0]

    def _u16(self, pos: int) -> int:
        return struct.unpack_from("<H", self.buf, pos)[0]

    def _u32(self, pos: int) -> int:
        return struct.unpack_from("<I", self.buf, pos)[0]

    def _i8(self, pos: int) -> int:
        return struct.unpack_from("<b", self.buf, pos)[0]

    def _i16(self, pos: int) -> int:
        return struct.unpack_from("<h", self.buf, pos)[0]

    def _i32(self, pos: int) -> int:
        return struct.unpack_from("<i", self.buf, pos)[0]

    def _f32(self, pos: int) -> float:
        return struct.unpack_from("<f", self.buf, pos)[0]

    # ------------------------------------------------------------------ #
    # Table / vtable helpers                                               #
    # ------------------------------------------------------------------ #

    def root(self) -> int:
        """Return the absolute position of the root table."""
        return self._u32(0)

    def _vtable_pos(self, table_pos: int) -> int:
        soffset = self._i32(table_pos)
        return table_pos - soffset

    def _field_offset(self, table_pos: int, field_id: int) -> int:
        """Return the within-table byte offset of field *field_id*, or 0 if absent."""
        vt = self._vtable_pos(table_pos)
        vt_size = self._u16(vt)
        slot = 4 + 2 * field_id
        if slot >= vt_size:
            return 0
        return self._u16(vt + slot)

    # ------------------------------------------------------------------ #
    # Scalar field accessors                                               #
    # ------------------------------------------------------------------ #

    def field_u8(self, tp: int, fid: int, default: int = 0) -> int:
        off = self._field_offset(tp, fid)
        return self._u8(tp + off) if off else default

    def field_i8(self, tp: int, fid: int, default: int = 0) -> int:
        off = self._field_offset(tp, fid)
        return self._i8(tp + off) if off else default

    def field_u16(self, tp: int, fid: int, default: int = 0) -> int:
        off = self._field_offset(tp, fid)
        return self._u16(tp + off) if off else default

    def field_i16(self, tp: int, fid: int, default: int = 0) -> int:
        off = self._field_offset(tp, fid)
        return self._i16(tp + off) if off else default

    def field_u32(self, tp: int, fid: int, default: int = 0) -> int:
        off = self._field_offset(tp, fid)
        return self._u32(tp + off) if off else default

    def field_i32(self, tp: int, fid: int, default: int = 0) -> int:
        off = self._field_offset(tp, fid)
        return self._i32(tp + off) if off else default

    def field_bool(self, tp: int, fid: int, default: bool = False) -> bool:
        return bool(self.field_u8(tp, fid, int(default)))

    # ------------------------------------------------------------------ #
    # String field accessor                                                #
    # ------------------------------------------------------------------ #

    def field_str(self, tp: int, fid: int, default: str = "") -> str:
        off = self._field_offset(tp, fid)
        if not off:
            return default
        abs_off = tp + off
        str_start = abs_off + self._u32(abs_off)
        length = self._u32(str_start)
        return self.buf[str_start + 4 : str_start + 4 + length].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Nested table field accessor                                          #
    # ------------------------------------------------------------------ #

    def field_table(self, tp: int, fid: int) -> int:
        """Return the absolute position of a nested table, or -1 if absent."""
        off = self._field_offset(tp, fid)
        if not off:
            return -1
        abs_off = tp + off
        return abs_off + self._u32(abs_off)

    # ------------------------------------------------------------------ #
    # Vector field accessor                                                #
    # ------------------------------------------------------------------ #

    def field_vector(self, tp: int, fid: int) -> int:
        """Return the absolute position of the vector header (length field), or -1 if absent."""
        off = self._field_offset(tp, fid)
        if not off:
            return -1
        abs_off = tp + off
        return abs_off + self._u32(abs_off)

    # ------------------------------------------------------------------ #
    # Vector element readers (vec_pos = position of the length field)      #
    # ------------------------------------------------------------------ #

    def vec_len(self, vec_pos: int) -> int:
        return self._u32(vec_pos)

    def vec_u8(self, vec_pos: int, idx: int) -> int:
        return self._u8(vec_pos + 4 + idx)

    def vec_i8(self, vec_pos: int, idx: int) -> int:
        return self._i8(vec_pos + 4 + idx)

    def vec_i16(self, vec_pos: int, idx: int) -> int:
        return self._i16(vec_pos + 4 + 2 * idx)

    def vec_u32(self, vec_pos: int, idx: int) -> int:
        return self._u32(vec_pos + 4 + 4 * idx)

    def vec_i32(self, vec_pos: int, idx: int) -> int:
        return self._i32(vec_pos + 4 + 4 * idx)

    def vec_table(self, vec_pos: int, idx: int) -> int:
        """Return the absolute position of table at *idx* in a vector of tables."""
        elem_pos = vec_pos + 4 + 4 * idx
        return elem_pos + self._u32(elem_pos)

    def vec_bytes(self, vec_pos: int) -> bytes:
        length = self._u32(vec_pos)
        return bytes(self.buf[vec_pos + 4 : vec_pos + 4 + length])


# ---------------------------------------------------------------------------
# TFLite built-in operator codes (subset used by the converter)
# ---------------------------------------------------------------------------


class BuiltinOperator:
    """Subset of the TFLite ``BuiltinOperator`` enum."""

    ADD = 0
    AVERAGE_POOL_2D = 1
    CONCATENATION = 2
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 4
    DEQUANTIZE = 6
    FLOOR = 8
    FULLY_CONNECTED = 9
    MAX_POOL_2D = 17
    MUL = 18
    RELU = 19
    RELU_N1_TO_1 = 20
    RESHAPE = 22
    SOFTMAX = 25
    TANH = 28
    PAD = 34
    TRANSPOSE = 39
    MEAN = 40
    SUB = 41
    DIV = 42
    SQUEEZE = 43
    EXP = 47
    LOG_SOFTMAX = 50
    NEG = 59
    SIN = 66
    TRANSPOSE_CONV = 67
    EXPAND_DIMS = 70
    LOG = 73
    SUM = 74
    SQRT = 75
    RSQRT = 76
    POW = 78
    REDUCE_MAX = 82
    LOGICAL_OR = 84
    LOGICAL_AND = 86
    LOGICAL_NOT = 87
    REDUCE_MIN = 89
    FLOOR_DIV = 90
    ABS = 101
    CEIL = 104
    LEAKY_RELU = 98
    SQUARED_DIFFERENCE = 99
    ELU = 111
    ROUND = 116
    HARD_SWISH = 117
    BATCH_MATMUL = 126
    GELU = 150

    CUSTOM = 32


# Reverse map: int → name
_BUILTIN_OP_NAMES: Dict[int, str] = {
    v: k for k, v in BuiltinOperator.__dict__.items() if isinstance(v, int)
}


def builtin_op_name(code: int) -> str:
    """Return the name of a TFLite ``BuiltinOperator`` code, e.g. ``'RELU'``."""
    return _BUILTIN_OP_NAMES.get(code, f"UNKNOWN({code})")


# ---------------------------------------------------------------------------
# TFLite fused activation function codes
# ---------------------------------------------------------------------------


class ActivationFunctionType:
    NONE = 0
    RELU = 1
    RELU_N1_TO_1 = 2
    RELU6 = 3
    TANH = 4
    SIGN_BIT = 5


# ---------------------------------------------------------------------------
# TFLite padding codes
# ---------------------------------------------------------------------------


class Padding:
    SAME = 0
    VALID = 1


# ---------------------------------------------------------------------------
# Parsed model dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TFLiteTensor:
    """Parsed representation of a TFLite :class:`Tensor`."""

    index: int
    name: str
    dtype: int  # TFLite TensorType int
    shape: Tuple[int, ...]  # static shape; dynamic dims encoded as -1
    data: Optional[np.ndarray]  # None if no buffer data (e.g., activation tensors)


@dataclass
class TFLiteOperator:
    """Parsed representation of a TFLite :class:`Operator`.

    The ``builtin_options`` dict contains decoded attributes for the
    current opcode (e.g. ``{"fused_activation": 0}`` for FULLY_CONNECTED).
    Unrecognised op options are left unparsed (empty dict).
    """

    opcode: int  # BuiltinOperator int; -1 means CUSTOM
    custom_code: str  # non-empty only when opcode == BuiltinOperator.CUSTOM
    inputs: Tuple[int, ...]  # tensor indices (-1 = optional absent)
    outputs: Tuple[int, ...]  # tensor indices
    builtin_options: Dict  # decoded option fields

    @property
    def name(self) -> str:
        """Human-readable op name for diagnostics."""
        if self.opcode == BuiltinOperator.CUSTOM:
            return f"CUSTOM({self.custom_code})"
        return builtin_op_name(self.opcode)


@dataclass
class TFLiteSubgraph:
    """Parsed representation of a TFLite :class:`SubGraph`."""

    name: str
    tensors: List[TFLiteTensor]
    inputs: Tuple[int, ...]  # indices into tensors
    outputs: Tuple[int, ...]  # indices into tensors
    operators: List[TFLiteOperator]


@dataclass
class TFLiteModel:
    """Top-level parsed TFLite model."""

    version: int
    subgraphs: List[TFLiteSubgraph] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FlatBuffer-based TFLite parser
# ---------------------------------------------------------------------------


def _read_int32_vec(fb: _FlatBuf, vec_pos: int) -> Tuple[int, ...]:
    if vec_pos < 0:
        return ()
    return tuple(fb.vec_i32(vec_pos, i) for i in range(fb.vec_len(vec_pos)))


def _parse_tensor(fb: _FlatBuf, buffers: List[Optional[bytes]], tp: int, idx: int) -> TFLiteTensor:
    """Parse a single TFLite Tensor table at *tp*."""
    # field 0: shape ([int32])
    shape_vec = fb.field_vector(tp, 0)
    if shape_vec >= 0:
        shape: Tuple[int, ...] = tuple(
            fb.vec_i32(shape_vec, i) for i in range(fb.vec_len(shape_vec))
        )
    else:
        shape = ()

    # field 1: type (int8 TensorType)
    dtype_int = fb.field_i8(tp, 1, default=0)

    # field 2: buffer index (uint32)
    buf_idx = fb.field_u32(tp, 2, default=0)

    # field 3: name (string)
    name = fb.field_str(tp, 3, default=f"tensor_{idx}")

    # Resolve buffer data to numpy array (if present)
    data: Optional[np.ndarray] = None
    if 0 < buf_idx < len(buffers) and buffers[buf_idx] is not None:
        raw = buffers[buf_idx]
        assert raw is not None
        try:
            np_dtype = litert_dtype_to_np_dtype(dtype_int)
            data = np.frombuffer(raw, dtype=np_dtype).reshape(shape) if shape else np.frombuffer(
                raw, dtype=np_dtype
            )
        except (ValueError, TypeError):
            # Unsupported dtype or reshape mismatch — leave data as None.
            data = None

    return TFLiteTensor(index=idx, name=name, dtype=dtype_int, shape=shape, data=data)


# ---------------------------------------------------------------------------
# Builtin options decoders (one per opcode, added incrementally)
# ---------------------------------------------------------------------------


def _decode_fully_connected_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    """Decode FullyConnectedOptions."""
    return {
        "fused_activation": fb.field_i8(opts_tp, 0, 0),  # ActivationFunctionType
        "weights_format": fb.field_i8(opts_tp, 1, 0),
        "keep_num_dims": fb.field_bool(opts_tp, 2, False),
    }


def _decode_conv2d_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    """Decode Conv2DOptions."""
    return {
        "padding": fb.field_i8(opts_tp, 0, 0),
        "stride_w": fb.field_i32(opts_tp, 1, 1),
        "stride_h": fb.field_i32(opts_tp, 2, 1),
        "fused_activation": fb.field_i8(opts_tp, 3, 0),
        "dilation_w_factor": fb.field_i32(opts_tp, 4, 1),
        "dilation_h_factor": fb.field_i32(opts_tp, 5, 1),
    }


def _decode_depthwise_conv2d_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    """Decode DepthwiseConv2DOptions."""
    return {
        "padding": fb.field_i8(opts_tp, 0, 0),
        "stride_w": fb.field_i32(opts_tp, 1, 1),
        "stride_h": fb.field_i32(opts_tp, 2, 1),
        "depth_multiplier": fb.field_i32(opts_tp, 3, 1),
        "fused_activation": fb.field_i8(opts_tp, 4, 0),
        "dilation_w_factor": fb.field_i32(opts_tp, 5, 1),
        "dilation_h_factor": fb.field_i32(opts_tp, 6, 1),
    }


def _decode_pool2d_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    """Decode Pool2DOptions (AvgPool and MaxPool share the same table)."""
    return {
        "padding": fb.field_i8(opts_tp, 0, 0),
        "stride_w": fb.field_i32(opts_tp, 1, 1),
        "stride_h": fb.field_i32(opts_tp, 2, 1),
        "filter_width": fb.field_i32(opts_tp, 3, 0),
        "filter_height": fb.field_i32(opts_tp, 4, 0),
        "fused_activation": fb.field_i8(opts_tp, 5, 0),
    }


def _decode_softmax_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    return {"beta": fb.field_u32(opts_tp, 0)}  # stored as float32 bits (f32 field)


def _decode_reshape_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    """Decode ReshapeOptions — only has a new_shape vector."""
    vec = fb.field_vector(opts_tp, 0)
    new_shape = _read_int32_vec(fb, vec) if vec >= 0 else ()
    return {"new_shape": new_shape}


def _decode_reduce_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    """Decode ReducerOptions (MEAN, SUM, REDUCE_MAX, etc.)."""
    return {"keep_dims": fb.field_bool(opts_tp, 0, False)}


def _decode_add_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    return {"fused_activation": fb.field_i8(opts_tp, 0, 0)}


def _decode_mul_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    return {"fused_activation": fb.field_i8(opts_tp, 0, 0)}


def _decode_sub_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    return {"fused_activation": fb.field_i8(opts_tp, 0, 0)}


def _decode_div_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    return {"fused_activation": fb.field_i8(opts_tp, 0, 0)}


def _decode_leaky_relu_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    alpha_bits = fb.field_u32(opts_tp, 0, 0)
    alpha = struct.unpack("<f", struct.pack("<I", alpha_bits))[0]
    return {"alpha": float(alpha)}


def _decode_concatenation_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    return {
        "axis": fb.field_i32(opts_tp, 0, 0),
        "fused_activation": fb.field_i8(opts_tp, 1, 0),
    }


def _decode_transpose_conv_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    return {
        "padding": fb.field_i8(opts_tp, 0, 0),
        "stride_w": fb.field_i32(opts_tp, 1, 1),
        "stride_h": fb.field_i32(opts_tp, 2, 1),
        "fused_activation": fb.field_i8(opts_tp, 3, 0),
    }


def _decode_batch_matmul_options(fb: _FlatBuf, opts_tp: int) -> Dict:
    return {
        "adj_x": fb.field_bool(opts_tp, 0, False),
        "adj_y": fb.field_bool(opts_tp, 1, False),
    }


# Maps BuiltinOperator code → (builtin_options_type, decoder_function)
# The builtin_options_type is the union discriminant stored in Operator.field 3.
_OPTION_DECODERS = {
    BuiltinOperator.CONV_2D: _decode_conv2d_options,
    BuiltinOperator.DEPTHWISE_CONV_2D: _decode_depthwise_conv2d_options,
    BuiltinOperator.AVERAGE_POOL_2D: _decode_pool2d_options,
    BuiltinOperator.MAX_POOL_2D: _decode_pool2d_options,
    BuiltinOperator.FULLY_CONNECTED: _decode_fully_connected_options,
    BuiltinOperator.SOFTMAX: _decode_softmax_options,
    BuiltinOperator.RESHAPE: _decode_reshape_options,
    BuiltinOperator.ADD: _decode_add_options,
    BuiltinOperator.MUL: _decode_mul_options,
    BuiltinOperator.SUB: _decode_sub_options,
    BuiltinOperator.DIV: _decode_div_options,
    BuiltinOperator.MEAN: _decode_reduce_options,
    BuiltinOperator.SUM: _decode_reduce_options,
    BuiltinOperator.REDUCE_MAX: _decode_reduce_options,
    BuiltinOperator.REDUCE_MIN: _decode_reduce_options,
    BuiltinOperator.LEAKY_RELU: _decode_leaky_relu_options,
    BuiltinOperator.CONCATENATION: _decode_concatenation_options,
    BuiltinOperator.TRANSPOSE_CONV: _decode_transpose_conv_options,
    BuiltinOperator.BATCH_MATMUL: _decode_batch_matmul_options,
}


def _parse_operator(
    fb: _FlatBuf,
    op_codes: List[Tuple[int, str]],
    tp: int,
) -> TFLiteOperator:
    """Parse a single TFLite Operator table at *tp*."""
    # field 0: opcode_index (uint32)
    opcode_idx = fb.field_u32(tp, 0, 0)
    builtin_code, custom_code = op_codes[opcode_idx] if opcode_idx < len(op_codes) else (-1, "")

    # field 1: inputs ([int32])
    inputs_vec = fb.field_vector(tp, 1)
    inputs = _read_int32_vec(fb, inputs_vec) if inputs_vec >= 0 else ()

    # field 2: outputs ([int32])
    outputs_vec = fb.field_vector(tp, 2)
    outputs = _read_int32_vec(fb, outputs_vec) if outputs_vec >= 0 else ()

    # field 4: builtin_options (union table)
    opts_tp = fb.field_table(tp, 4)
    builtin_options: Dict = {}
    if opts_tp >= 0 and builtin_code in _OPTION_DECODERS:
        try:
            builtin_options = _OPTION_DECODERS[builtin_code](fb, opts_tp)
        except (struct.error, IndexError):
            builtin_options = {}

    return TFLiteOperator(
        opcode=builtin_code,
        custom_code=custom_code,
        inputs=inputs,
        outputs=outputs,
        builtin_options=builtin_options,
    )


def parse_tflite_model(model: "str | os.PathLike | bytes") -> TFLiteModel:
    """Parse a ``.tflite`` file (or raw bytes) and return a :class:`TFLiteModel`.

    :param model: path to a ``.tflite`` file or its raw bytes
    :return: parsed :class:`TFLiteModel`
    """
    if isinstance(model, (str, os.PathLike)):
        with open(model, "rb") as fh:
            raw = fh.read()
    else:
        raw = bytes(model)

    fb = _FlatBuf(raw)
    root = fb.root()

    # Validate file identifier (optional but helps catch wrong inputs).
    if len(raw) >= 8 and raw[4:8] not in (
        _TFLITE_IDENTIFIER,
        b"TFL2",
        b"TFL1",
        _TFLITE_BLANK_IDENTIFIER,
    ):
        import warnings

        warnings.warn(
            f"Unexpected TFLite file identifier {raw[4:8]!r}. "
            "Proceeding anyway but the model may be corrupted.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------ #
    # Model.version (field 0, uint32)                                     #
    # ------------------------------------------------------------------ #
    version = fb.field_u32(root, 0, default=3)

    # ------------------------------------------------------------------ #
    # Model.buffers (field 4, [Buffer])                                   #
    # ------------------------------------------------------------------ #
    # We read buffers first so tensors can reference them.
    buffers_vec = fb.field_vector(root, 4)
    raw_buffers: List[Optional[bytes]] = []
    if buffers_vec >= 0:
        for bi in range(fb.vec_len(buffers_vec)):
            buf_tp = fb.vec_table(buffers_vec, bi)
            data_vec = fb.field_vector(buf_tp, 0)
            if data_vec >= 0 and fb.vec_len(data_vec) > 0:
                raw_buffers.append(fb.vec_bytes(data_vec))
            else:
                raw_buffers.append(None)

    # ------------------------------------------------------------------ #
    # Model.operator_codes (field 1, [OperatorCode])                      #
    # ------------------------------------------------------------------ #
    op_codes_vec = fb.field_vector(root, 1)
    op_codes: List[Tuple[int, str]] = []  # [(builtin_code, custom_code)]
    if op_codes_vec >= 0:
        for oci in range(fb.vec_len(op_codes_vec)):
            oc_tp = fb.vec_table(op_codes_vec, oci)
            # field 0: builtin_code (int8/BuiltinOperator, deprecated for > 127)
            builtin_i8 = fb.field_i8(oc_tp, 0, default=0)
            # field 4: extended_builtin_code (int32; use if available)
            extended = fb.field_i32(oc_tp, 4, default=0)
            builtin_code = extended if extended > 0 else int(builtin_i8)
            # field 2: custom_code (string)
            custom_code = fb.field_str(oc_tp, 2, default="")
            op_codes.append((builtin_code, custom_code))

    # ------------------------------------------------------------------ #
    # Model.subgraphs (field 2, [SubGraph])                               #
    # ------------------------------------------------------------------ #
    subgraphs_vec = fb.field_vector(root, 2)
    subgraphs: List[TFLiteSubgraph] = []
    if subgraphs_vec >= 0:
        for si in range(fb.vec_len(subgraphs_vec)):
            sg_tp = fb.vec_table(subgraphs_vec, si)
            sg_name = fb.field_str(sg_tp, 4, default=f"subgraph_{si}")

            # Tensors (field 0)
            tensors_vec = fb.field_vector(sg_tp, 0)
            tensors: List[TFLiteTensor] = []
            if tensors_vec >= 0:
                for ti in range(fb.vec_len(tensors_vec)):
                    t_tp = fb.vec_table(tensors_vec, ti)
                    tensors.append(_parse_tensor(fb, raw_buffers, t_tp, ti))

            # Inputs (field 1)
            inputs_vec_sg = fb.field_vector(sg_tp, 1)
            sg_inputs = _read_int32_vec(fb, inputs_vec_sg) if inputs_vec_sg >= 0 else ()

            # Outputs (field 2)
            outputs_vec_sg = fb.field_vector(sg_tp, 2)
            sg_outputs = _read_int32_vec(fb, outputs_vec_sg) if outputs_vec_sg >= 0 else ()

            # Operators (field 3)
            ops_vec = fb.field_vector(sg_tp, 3)
            operators: List[TFLiteOperator] = []
            if ops_vec >= 0:
                for oi in range(fb.vec_len(ops_vec)):
                    op_tp = fb.vec_table(ops_vec, oi)
                    operators.append(_parse_operator(fb, op_codes, op_tp))

            subgraphs.append(
                TFLiteSubgraph(
                    name=sg_name,
                    tensors=tensors,
                    inputs=sg_inputs,
                    outputs=sg_outputs,
                    operators=operators,
                )
            )

    return TFLiteModel(version=version, subgraphs=subgraphs)
