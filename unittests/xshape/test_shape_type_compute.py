import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape.shape_builder import ShapeBuilder
from yobx.xshape.shape_builder_impl import BasicShapeBuilder
from yobx.xshape.shape_type_compute import (
    broadcast_shape,
    set_type_shape_binary_op,
    set_type_shape_fused_matmul,
    set_type_shape_gemm,
    set_type_shape_matmul,
    set_type_shape_multi_head_attention,
    set_type_shape_reduce_op,
    set_type_shape_reshape,
    set_type_shape_scatter_nd_of_shape,
    set_type_shape_to_complex,
    set_type_shape_complex_module,
    set_type_shape_shared_input,
    set_type_shape_transpose_2d_cast_fp16,
    set_type_shape_transpose_2d_cast_fp32,
    set_type_shape_tree_ensemble,
    set_type_shape_tri_matrix,
    set_type_shape_unary_op,
    set_type_shape_unary_op_abs,
    set_shape_type_custom,
    _set_shape_type_op_any_sequence_empty,
    _set_shape_type_op_any_known,
    _set_shape_type_op_any_attention,
)

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16
TDOUBLE = onnx.TensorProto.DOUBLE
TINT64 = onnx.TensorProto.INT64
TBOOL = onnx.TensorProto.BOOL
TCOMPLEX64 = onnx.TensorProto.COMPLEX64
TCOMPLEX128 = onnx.TensorProto.COMPLEX128

_mkv_ = oh.make_tensor_value_info


class _TestShapeBuilder(BasicShapeBuilder):
    """BasicShapeBuilder extended with test-only helpers."""

    as_function = False
    _dim_counter = 0

    @property
    def torch(self):
        import torch

        return torch

    def unique_dimension_name(self, prefix: str) -> str:
        _TestShapeBuilder._dim_counter += 1
        return f"{prefix}_{_TestShapeBuilder._dim_counter}"

    def set_sequence(self, name: str, dtype: int = 0):
        self._known_types[name] = dtype

    def is_constant_or_attribute(
        self, node: onnx.NodeProto, input_index: int, attr_name: str
    ) -> bool:
        if input_index < len(node.input) and node.input[input_index]:
            return self.is_constant(node.input[input_index])
        for att in node.attribute:  # noqa: SIM110
            if att.name == attr_name:
                return True
        return False

    def get_constant_or_attribute(self, node: onnx.NodeProto, input_index: int, attr_name: str):
        if input_index < len(node.input) and node.input[input_index]:
            if self.is_constant(node.input[input_index]):
                return self.get_constant(node.input[input_index], computed_value=True)
        for att in node.attribute:
            if att.name == attr_name:
                if att.type == onnx.AttributeProto.INTS:
                    return list(att.ints)
                return att.i
        return None


def _make_model(nodes, inputs, outputs, initializers=None, opset=18):
    return oh.make_model(
        oh.make_graph(nodes, "test", inputs, outputs, initializers or []),
        opset_imports=[oh.make_opsetid("", opset)],
        ir_version=10,
    )


class _MockShapeBuilder(ShapeBuilder):
    """Minimal ShapeBuilder for unit-testing shape functions without torch."""

    def __init__(self):
        self._types = {}
        self._shapes = {}
        self._ranks = {}
        self._devices = {}
        self._debug_shape_missing = False

    def get_type(self, name):
        return self._types[name]

    def set_type(self, name, t):
        self._types[name] = t

    def has_type(self, name):
        return name in self._types

    def get_shape(self, name):
        return self._shapes[name]

    def set_shape(self, name, shape, allow_zero=False):
        self._shapes[name] = shape

    def has_shape(self, name):
        return name in self._shapes

    def get_rank(self, name):
        return self._ranks.get(name, len(self._shapes[name]) if name in self._shapes else None)

    def set_rank(self, name, rank):
        self._ranks[name] = rank

    def has_rank(self, name):
        return name in self._ranks or name in self._shapes

    def has_device(self, name):
        return name in self._devices

    def get_device(self, name):
        return self._devices[name]

    def set_device(self, name, d):
        self._devices[name] = d

    def get_debug_msg(self):
        return ""

    def register_constraint_dimension(self, d, v):
        pass

    @property
    def input_names(self):
        return []

    @property
    def output_names(self):
        return []


class TestShapeTypeCompute(ExtTestCase):
    # ------------------------------------------------------------------
    # broadcast_shape (already tested, kept for completeness)
    # ------------------------------------------------------------------

    def test_broadcast_shape_equal(self):
        self.assertEqual(broadcast_shape((3, 4), (3, 4)), (3, 4))

    def test_broadcast_shape_empty_first(self):
        self.assertEqual(broadcast_shape((), (3, 4)), (3, 4))

    def test_broadcast_shape_empty_second(self):
        self.assertEqual(broadcast_shape((3, 4), ()), (3, 4))

    def test_broadcast_shape_scalar_first(self):
        self.assertEqual(broadcast_shape((1,), (3, 4)), (3, 4))

    def test_broadcast_shape_scalar_second(self):
        self.assertEqual(broadcast_shape((3, 4), (1,)), (3, 4))

    def test_broadcast_shape_extend_rank(self):
        # (4,) broadcasts to (3, 4)
        self.assertEqual(broadcast_shape((4,), (3, 4)), (3, 4))

    def test_broadcast_shape_with_ones(self):
        self.assertEqual(broadcast_shape((1, 4), (3, 1)), (3, 4))

    def test_broadcast_shape_dynamic(self):
        result = broadcast_shape((1, "seq"), ("batch", "seq"))
        self.assertEqual(result, ("batch", "seq"))

    def test_broadcast_shape_zero(self):
        result = broadcast_shape((0, 4), (3, 4))
        self.assertEqual(result, (0, 4))

    def test_broadcast_shape_dynamic_both(self):
        result = broadcast_shape(("a", "b"), ("a", "b"))
        self.assertEqual(result, ("a", "b"))

    def test_broadcast_shape_int_overrides_one(self):
        # int=5 vs int=1 => 5
        self.assertEqual(broadcast_shape((5,), (1,)), (5,))
        self.assertEqual(broadcast_shape((1,), (5,)), (5,))

    # ------------------------------------------------------------------
    # set_type_shape_reshape
    # ------------------------------------------------------------------

    def test_set_type_shape_reshape_static(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        set_type_shape_reshape(b, "Y", "X", (2, 6))
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (2, 6))

    def test_set_type_shape_reshape_with_neg1(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        set_type_shape_reshape(b, "Y", "X", (6, -1))
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (6, 2))

    def test_set_type_shape_reshape_dynamic(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", ("a", "b"))
        # new_shape with string dimensions — only rank is set
        set_type_shape_reshape(b, "Y", "X", ("a", "b", 1))
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 3)

    def test_set_type_shape_reshape_new_shape_string_with_known_shape(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        # new_shape is a string and its shape is known: shape=(3,) → rank set to 3
        b.set_shape("shape_tensor", (3,))
        set_type_shape_reshape(b, "Y", "X", "shape_tensor")
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 3)

    def test_set_type_shape_reshape_new_shape_string_without_known_shape(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        # new_shape is a string but its shape is not registered → no rank/shape set
        set_type_shape_reshape(b, "Y", "X", "unknown_shape_tensor")
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertFalse(b.has_shape("Y"))
        self.assertFalse(b.has_rank("Y"))

    # ------------------------------------------------------------------
    # set_type_shape_unary_op
    # ------------------------------------------------------------------

    def test_set_type_shape_unary_op_with_shape(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        set_type_shape_unary_op(b, "Y", "X")
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_set_type_shape_unary_op_rank_only(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 2)
        set_type_shape_unary_op(b, "Y", "X")
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 2)

    def test_set_type_shape_unary_op_explicit_itype(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        set_type_shape_unary_op(b, "Y", "X", itype=TINT64)
        self.assertEqual(b.get_type("Y"), TINT64)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    # ------------------------------------------------------------------
    # set_type_shape_unary_op_abs
    # ------------------------------------------------------------------

    def test_set_type_shape_unary_op_abs_regular(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        set_type_shape_unary_op_abs(b, "Y", "X")
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_set_type_shape_unary_op_abs_complex64(self):
        b = BasicShapeBuilder()
        b.set_type("X", TCOMPLEX64)
        b.set_shape("X", (3, 4))
        set_type_shape_unary_op_abs(b, "Y", "X")
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_set_type_shape_unary_op_abs_complex128(self):
        b = BasicShapeBuilder()
        b.set_type("X", TCOMPLEX128)
        b.set_shape("X", (3, 4))
        set_type_shape_unary_op_abs(b, "Y", "X")
        self.assertEqual(b.get_type("Y"), TDOUBLE)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_set_type_shape_unary_op_abs_rank_only(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 3)
        set_type_shape_unary_op_abs(b, "Y", "X")
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 3)

    # ------------------------------------------------------------------
    # set_type_shape_binary_op
    # ------------------------------------------------------------------

    def test_set_type_shape_binary_op_same_shape(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (3, 4))
        set_type_shape_binary_op(b, "Z", "X", "Y")
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (3, 4))

    def test_set_type_shape_binary_op_broadcast(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (1, 4))
        set_type_shape_binary_op(b, "Z", "X", "Y")
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (3, 4))

    def test_set_type_shape_binary_op_cmp(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (3, 4))
        set_type_shape_binary_op(b, "Z", "X", "Y", cmp_op=True)
        self.assertEqual(b.get_type("Z"), TBOOL)
        self.assertEqual(b.get_shape("Z"), (3, 4))

    def test_set_type_shape_binary_op_explicit_itype(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (3, 4))
        set_type_shape_binary_op(b, "Z", "X", "Y", itype=TINT64)
        self.assertEqual(b.get_type("Z"), TINT64)

    def test_set_type_shape_binary_op_rank_only(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 3)
        b.set_type("Y", TFLOAT)
        b.set_rank("Y", 2)
        set_type_shape_binary_op(b, "Z", "X", "Y")
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_rank("Z"), 3)

    # ------------------------------------------------------------------
    # set_type_shape_matmul
    # ------------------------------------------------------------------

    def test_set_type_shape_matmul_2d(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (4, 5))
        result = set_type_shape_matmul(b, "Z", "X", "Y")
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (3, 5))
        self.assertEqual(result, (3, 5))

    def test_set_type_shape_matmul_batched(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (2, 3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (2, 4, 5))
        set_type_shape_matmul(b, "Z", "X", "Y")
        self.assertEqual(b.get_shape("Z"), (2, 3, 5))

    def test_set_type_shape_matmul_1d(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (4,))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (4,))
        set_type_shape_matmul(b, "Z", "X", "Y")
        self.assertEqual(b.get_shape("Z"), tuple())

    def test_set_type_shape_matmul_rank_only(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 3)
        b.set_type("Y", TFLOAT)
        b.set_rank("Y", 3)
        set_type_shape_matmul(b, "Z", "X", "Y")
        self.assertEqual(b.get_rank("Z"), 3)

    # ------------------------------------------------------------------
    # set_type_shape_gemm
    # ------------------------------------------------------------------

    def test_set_type_shape_gemm_no_trans(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (4, 5))
        set_type_shape_gemm(b, "Z", "X", "Y", transA=0, transB=0)
        self.assertEqual(b.get_shape("Z"), (3, 5))

    def test_set_type_shape_gemm_transA(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (4, 3))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (4, 5))
        set_type_shape_gemm(b, "Z", "X", "Y", transA=1, transB=0)
        self.assertEqual(b.get_shape("Z"), (3, 5))

    def test_set_type_shape_gemm_transB(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (5, 4))
        set_type_shape_gemm(b, "Z", "X", "Y", transA=0, transB=1)
        self.assertEqual(b.get_shape("Z"), (3, 5))

    # ------------------------------------------------------------------
    # set_type_shape_reduce_op
    # ------------------------------------------------------------------

    def test_set_type_shape_reduce_op_keepdim(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4, 5))
        set_type_shape_reduce_op(b, "Y", "X", keepdim=1, axes=(1,))
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 1, 5))

    def test_set_type_shape_reduce_op_no_keepdim(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4, 5))
        set_type_shape_reduce_op(b, "Y", "X", keepdim=0, axes=(1,))
        self.assertEqual(b.get_shape("Y"), (3, 5))

    def test_set_type_shape_reduce_op_no_axes(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4, 5))
        set_type_shape_reduce_op(b, "Y", "X", keepdim=1, axes=None)
        self.assertEqual(b.get_shape("Y"), (1, 1, 1))

    def test_set_type_shape_reduce_op_rank_only(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 3)
        set_type_shape_reduce_op(b, "Y", "X", keepdim=1, axes=(0,))
        self.assertEqual(b.get_rank("Y"), 3)

    # ------------------------------------------------------------------
    # _set_shape_type_op_any_* via ONNX models
    # ------------------------------------------------------------------

    def test_op_batch_normalization(self):
        model = _make_model(
            [oh.make_node("BatchNormalization", ["X", "sc", "bi", "mn", "vr"], ["Y"])],
            [_mkv_("X", TFLOAT, [2, 3, 4])],
            [_mkv_("Y", TFLOAT, [2, 3, 4])],
            [
                onh.from_array(np.ones(3, dtype=np.float32), name="sc"),
                onh.from_array(np.zeros(3, dtype=np.float32), name="bi"),
                onh.from_array(np.zeros(3, dtype=np.float32), name="mn"),
                onh.from_array(np.ones(3, dtype=np.float32), name="vr"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (2, 3, 4))

    def test_op_layer_normalization(self):
        model = _make_model(
            [oh.make_node("LayerNormalization", ["X", "sc", "bi"], ["Y"])],
            [_mkv_("X", TFLOAT, [2, 3, 4])],
            [_mkv_("Y", TFLOAT, [2, 3, 4])],
            [
                onh.from_array(np.ones(4, dtype=np.float32), name="sc"),
                onh.from_array(np.zeros(4, dtype=np.float32), name="bi"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (2, 3, 4))

    def test_op_cast(self):
        model = _make_model(
            [oh.make_node("Cast", ["X"], ["Y"], to=TINT64)],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TINT64, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TINT64)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_compress(self):
        model = _make_model(
            [oh.make_node("Compress", ["X", "cond"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4]), _mkv_("cond", TBOOL, [3])],
            [_mkv_("Y", TFLOAT, [None, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)

    def test_op_concat(self):
        model = _make_model(
            [oh.make_node("Concat", ["X", "Y"], ["Z"], axis=0)],
            [_mkv_("X", TFLOAT, [3, 4]), _mkv_("Y", TFLOAT, [2, 4])],
            [_mkv_("Z", TFLOAT, [5, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (5, 4))

    def test_op_conv(self):
        model = _make_model(
            [oh.make_node("Conv", ["X", "W"], ["Y"])],
            [_mkv_("X", TFLOAT, [1, 1, 8, 8])],
            [_mkv_("Y", TFLOAT, [1, 1, 6, 6])],
            [onh.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 1, 6, 6))

    def test_op_max_pool(self):
        model = _make_model(
            [oh.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2])],
            [_mkv_("X", TFLOAT, [1, 1, 6, 6])],
            [_mkv_("Y", TFLOAT, [1, 1, 5, 5])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 1, 5, 5))

    def test_op_conv_stride2(self):
        model = _make_model(
            [oh.make_node("Conv", ["X", "W"], ["Y"], strides=[2, 2])],
            [_mkv_("X", TFLOAT, [1, 1, 8, 8])],
            [_mkv_("Y", TFLOAT, [1, 1, 3, 3])],
            [onh.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 1, 3, 3))

    def test_op_max_pool_stride2(self):
        model = _make_model(
            [oh.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2])],
            [_mkv_("X", TFLOAT, [1, 1, 8, 8])],
            [_mkv_("Y", TFLOAT, [1, 1, 4, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 1, 4, 4))

    def test_op_gather(self):
        model = _make_model(
            [oh.make_node("Gather", ["X", "idx"], ["Y"], axis=0)],
            [_mkv_("X", TFLOAT, [5, 4]), _mkv_("idx", TINT64, [3])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 2)

    def test_op_gather_elements(self):
        model = _make_model(
            [oh.make_node("GatherElements", ["X", "idx"], ["Y"], axis=0)],
            [_mkv_("X", TFLOAT, [3, 4]), _mkv_("idx", TINT64, [2, 4])],
            [_mkv_("Y", TFLOAT, [2, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (2, 4))

    def test_op_gemm(self):
        model = _make_model(
            [oh.make_node("Gemm", ["X", "W"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 5])],
            [onh.from_array(np.ones((4, 5), dtype=np.float32), name="W")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 5))

    def test_op_matmul(self):
        model = _make_model(
            [oh.make_node("MatMul", ["X", "Y"], ["Z"])],
            [_mkv_("X", TFLOAT, [3, 4]), _mkv_("Y", TFLOAT, [4, 5])],
            [_mkv_("Z", TFLOAT, [3, 5])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (3, 5))

    def test_op_non_zero(self):
        model = _make_model(
            [oh.make_node("NonZero", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TINT64, [2, None])],
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TINT64)
        # NonZero output shape: (rank_of_input, num_nonzero)
        self.assertEqual(b.get_shape("Y")[0], 2)

    def test_op_pad(self):
        model = _make_model(
            [oh.make_node("Pad", ["X", "pads"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [5, 6])],
            [onh.from_array(np.array([1, 1, 1, 1], dtype=np.int64), name="pads")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (5, 6))

    def test_op_range(self):
        model = _make_model(
            [oh.make_node("Range", ["start", "limit", "delta"], ["Y"])],
            [],
            [_mkv_("Y", TINT64, [None])],
            [
                onh.from_array(np.array(0, dtype=np.int64), name="start"),
                onh.from_array(np.array(5, dtype=np.int64), name="limit"),
                onh.from_array(np.array(1, dtype=np.int64), name="delta"),
            ],
        )
        # Use _TestShapeBuilder (which has unique_dimension_name) because
        # scalar INT64 constants are not treated as known shape values so
        # the dim is dynamic.
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TINT64)
        self.assertEqual(b.get_rank("Y"), 1)

    def test_op_range_known_values(self):
        """Range with known value shapes computes the exact output size."""
        model = _make_model(
            [oh.make_node("Range", ["start", "limit", "delta"], ["Y"])],
            [],
            [_mkv_("Y", TINT64, [None])],
            [
                onh.from_array(np.array(0, dtype=np.int64), name="start"),
                onh.from_array(np.array(5, dtype=np.int64), name="limit"),
                onh.from_array(np.array(1, dtype=np.int64), name="delta"),
            ],
        )
        b = _TestShapeBuilder()
        # Manually register value shapes so the range size can be computed.
        b._known_value_shape["start"] = 0
        b._known_value_shape["limit"] = 5
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TINT64)
        self.assertEqual(b.get_shape("Y"), (5,))

    def test_op_reduce_sum(self):
        model = _make_model(
            [oh.make_node("ReduceSum", ["X", "axes"], ["Y"], keepdims=1)],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 1])],
            [onh.from_array(np.array([1], dtype=np.int64), name="axes")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 1))

    def test_op_reshape(self):
        model = _make_model(
            [oh.make_node("Reshape", ["X", "shape"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [2, 6])],
            [onh.from_array(np.array([2, 6], dtype=np.int64), name="shape")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (2, 6))

    def test_op_expand(self):
        model = _make_model(
            [oh.make_node("Expand", ["X", "shape"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 1])],
            [_mkv_("Y", TFLOAT, [3, 4])],
            [onh.from_array(np.array([3, 4], dtype=np.int64), name="shape")],
        )
        # _TestShapeBuilder has a 'torch' property needed by _make_node_set_type_shape_constant
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_expand_with_one(self):
        # value is not None and cst contains 1, so _apply_expand_to_shape is called
        model = _make_model(
            [oh.make_node("Expand", ["X", "shape"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 1])],
            [_mkv_("Y", TFLOAT, [3, 4])],
            [onh.from_array(np.array([1, 4], dtype=np.int64), name="shape")],
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_sign(self):
        model = _make_model(
            [oh.make_node("Sign", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_slice(self):
        model = _make_model(
            [oh.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])],
            [_mkv_("X", TFLOAT, [5, 6])],
            [_mkv_("Y", TFLOAT, [None, None])],
            [
                onh.from_array(np.array([1], dtype=np.int64), name="starts"),
                onh.from_array(np.array([4], dtype=np.int64), name="ends"),
                onh.from_array(np.array([0], dtype=np.int64), name="axes"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 2)

    def test_op_split(self):
        model = _make_model(
            [oh.make_node("Split", ["X", "sp"], ["A", "B"], axis=0)],
            [_mkv_("X", TFLOAT, [6, 4])],
            [_mkv_("A", TFLOAT, [3, 4]), _mkv_("B", TFLOAT, [3, 4])],
            [onh.from_array(np.array([3, 3], dtype=np.int64), name="sp")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("A"), TFLOAT)
        self.assertEqual(b.get_shape("A"), (3, 4))
        self.assertEqual(b.get_shape("B"), (3, 4))

    def test_op_scatternd(self):
        model = _make_model(
            [oh.make_node("ScatterND", ["X", "idx", "upd"], ["Y"])],
            [
                _mkv_("X", TFLOAT, [4, 4]),
                _mkv_("idx", TINT64, [2, 1]),
                _mkv_("upd", TFLOAT, [2, 4]),
            ],
            [_mkv_("Y", TFLOAT, [4, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (4, 4))

    def test_op_sequence_empty(self):
        node = oh.make_node("SequenceEmpty", [], ["Y"], dtype=TFLOAT)
        b = _TestShapeBuilder()
        result = _set_shape_type_op_any_sequence_empty(b, node)
        self.assertTrue(result)
        self.assertEqual(b.get_type("Y"), TFLOAT)

    def test_op_transpose(self):
        model = _make_model(
            [oh.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [_mkv_("X", TFLOAT, [2, 3, 4])],
            [_mkv_("Y", TFLOAT, [3, 2, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 2, 4))

    def test_op_tile(self):
        model = _make_model(
            [oh.make_node("Tile", ["X", "reps"], ["Y"])],
            [_mkv_("X", TFLOAT, [2, 3])],
            [_mkv_("Y", TFLOAT, [4, 9])],
            [onh.from_array(np.array([2, 3], dtype=np.int64), name="reps")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 2)

    def test_op_topk(self):
        model = _make_model(
            [oh.make_node("TopK", ["X", "k"], ["vals", "idx"])],
            [_mkv_("X", TFLOAT, [3, 10])],
            [_mkv_("vals", TFLOAT, [3, 5]), _mkv_("idx", TINT64, [3, 5])],
            [onh.from_array(np.array([5], dtype=np.int64), name="k")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("vals"), TFLOAT)
        self.assertEqual(b.get_shape("vals"), (3, 5))
        self.assertEqual(b.get_type("idx"), TINT64)
        self.assertEqual(b.get_shape("idx"), (3, 5))

    def test_op_unsqueeze(self):
        model = _make_model(
            [oh.make_node("Unsqueeze", ["X", "axes"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [1, 3, 4])],
            [onh.from_array(np.array([0], dtype=np.int64), name="axes")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 3, 4))

    def test_op_squeeze(self):
        model = _make_model(
            [oh.make_node("Squeeze", ["X", "axes"], ["Y"])],
            [_mkv_("X", TFLOAT, [1, 3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
            [onh.from_array(np.array([0], dtype=np.int64), name="axes")],
        )
        # _TestShapeBuilder has is_constant_or_attribute used by value-shape tracking
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_where(self):
        model = _make_model(
            [oh.make_node("Where", ["cond", "X", "Y"], ["Z"])],
            [
                _mkv_("cond", TBOOL, [3, 4]),
                _mkv_("X", TFLOAT, [3, 4]),
                _mkv_("Y", TFLOAT, [3, 4]),
            ],
            [_mkv_("Z", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (3, 4))

    def test_op_gelu(self):
        model = _make_model(
            [oh.make_node("Gelu", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_log(self):
        model = _make_model(
            [oh.make_node("Log", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_add_element_wise(self):
        model = _make_model(
            [oh.make_node("Add", ["X", "Y"], ["Z"])],
            [_mkv_("X", TFLOAT, [3, 4]), _mkv_("Y", TFLOAT, [3, 4])],
            [_mkv_("Z", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (3, 4))

    def test_op_equal_cmp(self):
        model = _make_model(
            [oh.make_node("Equal", ["X", "Y"], ["Z"])],
            [_mkv_("X", TINT64, [3, 4]), _mkv_("Y", TINT64, [3, 4])],
            [_mkv_("Z", TBOOL, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Z"), TBOOL)
        self.assertEqual(b.get_shape("Z"), (3, 4))

    def test_op_pow(self):
        model = _make_model(
            [oh.make_node("Pow", ["X", "Y"], ["Z"])],
            [_mkv_("X", TFLOAT, [3, 4]), _mkv_("Y", TFLOAT, [3, 4])],
            [_mkv_("Z", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (3, 4))

    def test_op_abs(self):
        model = _make_model(
            [oh.make_node("Abs", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_relu(self):
        model = _make_model(
            [oh.make_node("Relu", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_is_nan(self):
        model = _make_model(
            [oh.make_node("IsNaN", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TBOOL, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TBOOL)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_is_inf(self):
        model = _make_model(
            [oh.make_node("IsInf", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TBOOL, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TBOOL)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    # ------------------------------------------------------------------
    # set_type_shape_fused_matmul
    # ------------------------------------------------------------------

    def test_set_type_shape_fused_matmul_no_trans(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (4, 5))
        node = oh.make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft")
        set_type_shape_fused_matmul(b, node)
        self.assertEqual(b.get_shape("Z"), (3, 5))

    def test_set_type_shape_fused_matmul_transA(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (4, 3))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (4, 5))
        node = oh.make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft", transA=1)
        set_type_shape_fused_matmul(b, node)
        self.assertEqual(b.get_shape("Z"), (3, 5))

    def test_set_type_shape_fused_matmul_rank_only(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 2)
        b.set_type("Y", TFLOAT)
        b.set_rank("Y", 2)
        node = oh.make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft")
        set_type_shape_fused_matmul(b, node)
        self.assertEqual(b.get_rank("Z"), 2)

    # ------------------------------------------------------------------
    # set_type_shape_tree_ensemble
    # ------------------------------------------------------------------

    def test_set_type_shape_tree_ensemble(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (10, 5))
        node = oh.make_node(
            "TreeEnsembleRegressor",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            n_targets=3,
            nodes_falsenodeids=[0],
            nodes_featureids=[0],
            nodes_hitrates=[1.0],
            nodes_missing_value_tracks_true=[0],
            nodes_modes=["BRANCH_LEQ"],
            nodes_nodeids=[0],
            nodes_treeids=[0],
            nodes_truenodeids=[0],
            nodes_values=[0.0],
            post_transform="NONE",
            target_ids=[0],
            target_nodeids=[0],
            target_treeids=[0],
            target_weights=[1.0],
        )
        set_type_shape_tree_ensemble(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (10, 3))

    def test_set_type_shape_tree_ensemble_rank_only(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 2)
        node = oh.make_node(
            "TreeEnsembleRegressor",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            n_targets=2,
            nodes_falsenodeids=[0],
            nodes_featureids=[0],
            nodes_hitrates=[1.0],
            nodes_missing_value_tracks_true=[0],
            nodes_modes=["BRANCH_LEQ"],
            nodes_nodeids=[0],
            nodes_treeids=[0],
            nodes_truenodeids=[0],
            nodes_values=[0.0],
            post_transform="NONE",
            target_ids=[0],
            target_nodeids=[0],
            target_treeids=[0],
            target_weights=[1.0],
        )
        set_type_shape_tree_ensemble(b, node)
        self.assertEqual(b.get_rank("Y"), 2)

    # ------------------------------------------------------------------
    # set_type_shape_to_complex / set_type_shape_complex_module
    # ------------------------------------------------------------------

    def test_set_type_shape_to_complex_float(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4, 2))
        node = oh.make_node("ToComplex", ["X"], ["Y"], domain="com.microsoft")
        set_type_shape_to_complex(b, node)
        self.assertEqual(b.get_type("Y"), TCOMPLEX64)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_set_type_shape_to_complex_double(self):
        b = BasicShapeBuilder()
        b.set_type("X", TDOUBLE)
        b.set_shape("X", (3, 4, 2))
        node = oh.make_node("ToComplex", ["X"], ["Y"], domain="com.microsoft")
        set_type_shape_to_complex(b, node)
        self.assertEqual(b.get_type("Y"), TCOMPLEX128)

    def test_set_type_shape_complex_module(self):
        b = BasicShapeBuilder()
        b.set_type("X", TCOMPLEX64)
        b.set_shape("X", (3, 4))
        node = oh.make_node("ComplexModule", ["X"], ["Y"], domain="com.microsoft")
        set_type_shape_complex_module(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    # ------------------------------------------------------------------
    # set_type_shape_shared_input
    # ------------------------------------------------------------------

    def test_set_type_shape_shared_input(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (3, 4))
        b.set_type("Z", TFLOAT)
        b.set_shape("Z", (3, 4))
        node = oh.make_node(
            "AddSharedInput", ["X", "Y", "Z"], ["out1", "out2"], domain="com.microsoft"
        )
        set_type_shape_shared_input(b, node)
        self.assertEqual(b.get_type("out1"), TFLOAT)
        self.assertEqual(b.get_type("out2"), TFLOAT)

    # ------------------------------------------------------------------
    # set_type_shape_scatter_nd_of_shape
    # ------------------------------------------------------------------

    def test_set_type_shape_scatter_nd_of_shape(self):
        b = BasicShapeBuilder()
        b.set_type("shape", TINT64)
        b.set_shape("shape", (3,))
        b.constants_["shape"] = onh.from_array(np.array([2, 3, 4], dtype=np.int64))
        b.constants_computed_["shape"] = np.array([2, 3, 4], dtype=np.int64)
        b._known_value_shape["shape"] = (2, 3, 4)
        b.set_type("idx", TINT64)
        b.set_type("upd", TFLOAT)
        b.set_shape("upd", (5, 3, 4))
        node = oh.make_node(
            "ScatterNDOfShape",
            ["shape", "idx", "upd"],
            ["Y"],
            domain="com.microsoft",
        )
        set_type_shape_scatter_nd_of_shape(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (2, 3, 4))

    # ------------------------------------------------------------------
    # set_type_shape_tri_matrix
    # ------------------------------------------------------------------

    def test_set_type_shape_tri_matrix(self):
        b = BasicShapeBuilder()
        b.set_type("shape", TINT64)
        b.set_shape("shape", (2,))
        b.constants_["shape"] = onh.from_array(np.array([4, 4], dtype=np.int64))
        b.constants_computed_["shape"] = np.array([4, 4], dtype=np.int64)
        b._known_value_shape["shape"] = (4, 4)
        b.set_type("val", TFLOAT)
        b.set_shape("val", ())
        node = oh.make_node("TriMatrix", ["shape", "val"], ["Y"], domain="com.microsoft")
        set_type_shape_tri_matrix(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (4, 4))

    # ------------------------------------------------------------------
    # set_type_shape_transpose_2d_cast_fp16 / fp32
    # ------------------------------------------------------------------

    def test_set_type_shape_transpose_2d_cast_fp16(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        node = oh.make_node("Transpose2DCastFP16", ["X"], ["Y"], domain="com.microsoft")
        set_type_shape_transpose_2d_cast_fp16(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT16)
        self.assertEqual(b.get_shape("Y"), (4, 3))

    def test_set_type_shape_transpose_2d_cast_fp32(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT16)
        b.set_shape("X", (3, 4))
        node = oh.make_node("Transpose2DCastFP32", ["X"], ["Y"], domain="com.microsoft")
        set_type_shape_transpose_2d_cast_fp32(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (4, 3))

    # ------------------------------------------------------------------
    # set_type_shape_multi_head_attention
    # ------------------------------------------------------------------

    def test_set_type_shape_multi_head_attention(self):
        b = BasicShapeBuilder()
        for name in ("Q", "K", "V"):
            b.set_type(name, TFLOAT)
            b.set_shape(name, (2, 5, 64))
        b.set_type("pk", TFLOAT)
        b.set_shape("pk", (2, 1, 3, 64))
        node = oh.make_node(
            "MultiHeadAttention",
            ["Q", "K", "V", "", "", "", "pk"],
            ["out1", "out2"],
            domain="com.microsoft",
        )
        result = set_type_shape_multi_head_attention(b, node)
        self.assertIsNotNone(result)
        self.assertEqual(b.get_shape("out1"), (2, 5, 64))

    def test_set_type_shape_multi_head_attention_rank_only(self):
        b = BasicShapeBuilder()
        for name in ("Q", "K", "V"):
            b.set_type(name, TFLOAT)
            b.set_rank(name, 3)
        node = oh.make_node(
            "MultiHeadAttention",
            ["Q", "K", "V"],
            ["out1"],
            domain="com.microsoft",
        )
        set_type_shape_multi_head_attention(b, node)
        self.assertEqual(b.get_rank("out1"), 3)

    # ------------------------------------------------------------------
    # set_shape_type_custom
    # ------------------------------------------------------------------

    def test_set_shape_type_custom_replace_zero(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        node = oh.make_node("ReplaceZero", ["X"], ["Y"], domain="com.microsoft")
        set_shape_type_custom(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_set_shape_type_custom_fused_matmul(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (4, 5))
        node = oh.make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft")
        set_shape_type_custom(b, node)
        self.assertEqual(b.get_shape("Z"), (3, 5))

    def test_set_shape_type_custom_tree_ensemble(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (5, 3))
        node = oh.make_node(
            "TreeEnsembleRegressor",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            n_targets=2,
            nodes_falsenodeids=[0],
            nodes_featureids=[0],
            nodes_hitrates=[1.0],
            nodes_missing_value_tracks_true=[0],
            nodes_modes=["BRANCH_LEQ"],
            nodes_nodeids=[0],
            nodes_treeids=[0],
            nodes_truenodeids=[0],
            nodes_values=[0.0],
            post_transform="NONE",
            target_ids=[0],
            target_nodeids=[0],
            target_treeids=[0],
            target_weights=[1.0],
        )
        set_shape_type_custom(b, node)
        self.assertEqual(b.get_shape("Y"), (5, 2))

    def test_argmax_keepdims(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 4)
        node = oh.make_node("ArgMax", ["X"], ["Y"], axis=1, keepdims=1)
        _set_shape_type_op_any_known["ArgMax"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 1, 4))
        self.assertEqual(g._types.get("Y"), TINT64)

    def test_argmin_no_keepdims(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 4)
        node = oh.make_node("ArgMin", ["X"], ["Y"], axis=2, keepdims=0)
        _set_shape_type_op_any_known["ArgMin"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 3))
        self.assertEqual(g._types.get("Y"), TINT64)

    def test_global_average_pool(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 8, 4, 4)
        node = oh.make_node("GlobalAveragePool", ["X"], ["Y"])
        _set_shape_type_op_any_known["GlobalAveragePool"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 8, 1, 1))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_global_max_pool(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 16, 6, 6)
        node = oh.make_node("GlobalMaxPool", ["X"], ["Y"])
        _set_shape_type_op_any_known["GlobalMaxPool"](g, node)
        self.assertEqual(g._shapes.get("Y"), (1, 16, 1, 1))

    def test_flatten_static(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 4)
        node = oh.make_node("Flatten", ["X"], ["Y"], axis=1)
        _set_shape_type_op_any_known["Flatten"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 12))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_flatten_dynamic(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = ("batch", 3, 4)
        node = oh.make_node("Flatten", ["X"], ["Y"], axis=1)
        _set_shape_type_op_any_known["Flatten"](g, node)
        self.assertEqual(g._shapes.get("Y"), ("batch", 12))

    def test_eyelike_same_type(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (3, 3)
        node = oh.make_node("EyeLike", ["X"], ["Y"])
        _set_shape_type_op_any_known["EyeLike"](g, node)
        self.assertEqual(g._shapes.get("Y"), (3, 3))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_eyelike_with_dtype(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (4, 4)
        node = oh.make_node("EyeLike", ["X"], ["Y"], dtype=TINT64)
        _set_shape_type_op_any_known["EyeLike"](g, node)
        self.assertEqual(g._shapes.get("Y"), (4, 4))
        self.assertEqual(g._types.get("Y"), TINT64)

    def test_depth_to_space(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 8, 2, 3)
        node = oh.make_node("DepthToSpace", ["X"], ["Y"], blocksize=2)
        _set_shape_type_op_any_known["DepthToSpace"](g, node)
        self.assertEqual(g._shapes.get("Y"), (1, 2, 4, 6))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_space_to_depth(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 2, 4, 6)
        node = oh.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)
        _set_shape_type_op_any_known["SpaceToDepth"](g, node)
        self.assertEqual(g._shapes.get("Y"), (1, 8, 2, 3))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    # ------------------------------------------------------------------
    # _set_shape_type_op_any_attention
    # ------------------------------------------------------------------

    def test_set_shape_type_op_any_attention_4d(self):
        # 4D input: (batch, head, seq, size)
        g = _MockShapeBuilder()
        for name, shape in [("Q", (2, 8, 10, 64)), ("K", (2, 8, 10, 64)), ("V", (2, 8, 10, 32))]:
            g._types[name] = TFLOAT
            g._shapes[name] = shape
        node = oh.make_node("Attention", ["Q", "K", "V"], ["out"])
        _set_shape_type_op_any_attention(g, node)
        self.assertEqual(g._shapes.get("out"), (2, 8, 10, 32))
        self.assertEqual(g._types.get("out"), TFLOAT)

    def test_set_shape_type_op_any_attention_3d(self):
        # 3D input: (batch, seq, hidden_size) with q/kv head attributes
        g = _MockShapeBuilder()
        for name, shape in [
            ("Q", (2, 10, 512)),
            ("K", (2, 10, 256)),
            ("V", (2, 10, 256)),
        ]:
            g._types[name] = TFLOAT
            g._shapes[name] = shape
        node = oh.make_node("Attention", ["Q", "K", "V"], ["out"], q_num_heads=8, kv_num_heads=4)
        _set_shape_type_op_any_attention(g, node)
        # v_size = 256 // 4 = 64; output shape = (batch, seq, q_head * v_size) = (2, 10, 512)
        self.assertEqual(g._shapes.get("out"), (2, 10, 512))
        self.assertEqual(g._types.get("out"), TFLOAT)

    def test_set_shape_type_op_any_attention_3d_present_outputs(self):
        # 3D with present key/value outputs (output[1] and output[2]), no past inputs
        g = _MockShapeBuilder()
        for name, shape in [
            ("Q", (2, 10, 512)),
            ("K", (2, 10, 512)),
            ("V", (2, 10, 512)),
        ]:
            g._types[name] = TFLOAT
            g._shapes[name] = shape
        node = oh.make_node(
            "Attention",
            ["Q", "K", "V", "", ""],
            ["out", "present_key", "present_value"],
            q_num_heads=8,
            kv_num_heads=8,
        )
        _set_shape_type_op_any_attention(g, node)
        # q_head=8, k_head=v_head=8, k_size=v_size=64
        # output[0]: (2, 10, 8*64) = (2, 10, 512)
        self.assertEqual(g._shapes.get("out"), (2, 10, 512))
        # No past inputs: past_seq=0, total_seq=10+0=10
        # output[1]: (batch, k_head, total_seq, k_size) = (2, 8, 10, 64)
        self.assertEqual(g._shapes.get("present_key"), (2, 8, 10, 64))
        # output[2]: (batch, v_head, total_seq, v_size) = (2, 8, 10, 64)
        self.assertEqual(g._shapes.get("present_value"), (2, 8, 10, 64))

    def test_set_shape_type_op_any_attention_rank_only(self):
        # Only rank information available; rank of output[0] = rank of input[0]
        g = _MockShapeBuilder()
        g._types["Q"] = TFLOAT
        g._ranks["Q"] = 3
        node = oh.make_node("Attention", ["Q", "K", "V"], ["out"])
        _set_shape_type_op_any_attention(g, node)
        self.assertEqual(g._ranks.get("out"), 3)

    def test_set_shape_type_op_any_attention_type_propagation(self):
        # Type from input[0] propagates to all outputs; output[2] uses input[2] type
        g = _MockShapeBuilder()
        g._types["Q"] = TFLOAT16
        g._types["K"] = TFLOAT16
        g._types["V"] = TFLOAT  # different type for v cache output
        node = oh.make_node("Attention", ["Q", "K", "V"], ["out", "present_key", "present_value"])
        _set_shape_type_op_any_attention(g, node)
        self.assertEqual(g._types.get("out"), TFLOAT16)
        self.assertEqual(g._types.get("present_key"), TFLOAT16)
        self.assertEqual(g._types.get("present_value"), TFLOAT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
