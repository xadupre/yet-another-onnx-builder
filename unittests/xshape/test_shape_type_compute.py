import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape import ShapeBuilder, BasicShapeBuilder
from yobx.xshape.shape_type_compute import (
    broadcast_shape,
    compute_reshape_shape,
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
    supported_ops_in_set_shape_type_custom,
    set_shape_type_op_any_sequence_empty,
    set_shape_type_op_any_known,
    set_shape_type_op_any_attention,
    set_shape_type_op_any_loop,
    set_shape_type_op_any_squeeze,
    set_shape_type_op_any_unsqueeze,
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


class _LocalFunctionShapeBuilder(BasicShapeBuilder):
    """BasicShapeBuilder with local function support for testing set_shape_type_custom."""

    def __init__(self):
        super().__init__()
        self._functions = {}  # (name, domain) -> FunctionProto
        self._functions_builder = {}  # (name, domain) -> builder
        self._func_nodes = []  # nodes executed during infer_shapes

    @property
    def functions_builder(self):
        return self._functions_builder

    def has_local_function(self, op_type, domain="", builder=False):
        key = (op_type, domain)
        return key in (self._functions_builder if builder else self._functions)

    def get_local_function(self, op_type, domain="", builder=False):
        key = (op_type, domain)
        return (self._functions_builder if builder else self._functions).get(key)

    def register_local_function(self, proto, func_builder):
        """Register a FunctionProto together with its shape-inference builder."""
        key = (proto.name, proto.domain)
        self._functions[key] = proto
        self._functions_builder[key] = func_builder

    def reset_types_and_shapes(self):
        """Clear cached shapes and types so they can be recomputed."""
        self._known_shapes = {}
        self._known_types = {}
        self._known_ranks = {}
        self._known_devices = {}
        self.constants_ = {}
        self._calls = []

    def infer_shapes(self):
        """Re-run shape inference over the stored function nodes."""
        for node in self._func_nodes:
            self.run_node(node)


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

    def test_broadcast_shape_registers_constraint_str_vs_int(self):
        # When a symbolic dimension meets a non-1 integer, the integer wins
        # and a constraint is registered: symbolic_dim = concrete_int.
        # Here (64,) is right-aligned with ("batch", "d_model"):
        #   batch vs 1  -> batch (1 broadcasts to anything)
        #   d_model vs 64 -> 64 (concrete int wins), constraint d_model=64 recorded
        b = BasicShapeBuilder()
        result = broadcast_shape(("batch", "d_model"), (64,), graph_builder=b)
        self.assertEqual(result, ("batch", 64))
        constraints = b.get_registered_constraints()
        self.assertIn("d_model", constraints)
        self.assertIn(64, constraints["d_model"])

    def test_broadcast_shape_registers_constraint_int_vs_str(self):
        # Same scenario but concrete integer is in the first shape.
        b = BasicShapeBuilder()
        result = broadcast_shape((64,), ("batch", "d_model"), graph_builder=b)
        self.assertEqual(result, ("batch", 64))
        constraints = b.get_registered_constraints()
        self.assertIn("d_model", constraints)
        self.assertIn(64, constraints["d_model"])

    def test_broadcast_shape_no_constraint_when_int_is_one(self):
        # A concrete dimension of 1 broadcasts to the symbolic dimension;
        # no constraint should be registered in this case.
        b = BasicShapeBuilder()
        result = broadcast_shape(("batch", "seq"), (1, 1), graph_builder=b)
        self.assertEqual(result, ("batch", "seq"))
        self.assertEqual(b.get_registered_constraints(), {})

    def test_broadcast_shape_without_graph_builder_no_constraint_stored(self):
        # When no graph_builder is provided, the result is still correct but
        # no constraint object is available for inspection.
        result = broadcast_shape(("batch", "d_model"), (64,))
        self.assertEqual(result, ("batch", 64))

    def test_broadcast_shape_constraint_propagated_through_model(self):
        # Full end-to-end: broadcast a symbolic-shaped input against a
        # constant bias; verify both the output shape and the constraint.
        import onnx.numpy_helper as onh

        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "bias"], ["Z"])],
                "graph",
                [oh.make_tensor_value_info("X", TFLOAT, ["batch", "d_model"])],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
                [onh.from_array(np.zeros((64,), dtype=np.float32), name="bias")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        # The concrete size 64 wins over the symbolic "d_model".
        self.assertEqual(b.get_shape("Z"), ("batch", 64))
        # The constraint records the relationship d_model = 64.
        constraints = b.get_registered_constraints()
        self.assertIn("d_model", constraints)
        self.assertIn(64, constraints["d_model"])

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

    def test_set_type_shape_reshape_dynamic_input_neg1(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", ("N",))
        # 1-D input with one dynamic dim; reshape(-1, 1) → the -1 should be propagated
        set_type_shape_reshape(b, "Y", "X", (-1, 1))
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), ("N", 1))

    def test_set_type_shape_reshape_dynamic_input_neg1_fallback(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", ("N", "M"))
        # 2-D input with two dynamic dims; cannot resolve -1 → only rank set
        set_type_shape_reshape(b, "Y", "X", (-1, 3))
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 2)

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
    # LogSoftmax shape compute
    # ------------------------------------------------------------------

    def test_logsoftmax_with_shape(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (2, 3, 4))
        node = oh.make_node("LogSoftmax", inputs=["X"], outputs=["Y"], axis=1)
        set_shape_type_op_any_known["LogSoftmax"](b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (2, 3, 4))

    def test_logsoftmax_with_rank_only(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 3)
        node = oh.make_node("LogSoftmax", inputs=["X"], outputs=["Y"], axis=2)
        set_shape_type_op_any_known["LogSoftmax"](b, node)
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
    # set_shape_type_op_any_* via ONNX models
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

    def test_op_instance_normalization(self):
        model = _make_model(
            [oh.make_node("InstanceNormalization", ["X", "sc", "bi"], ["Y"])],
            [_mkv_("X", TFLOAT, [2, 3, 4])],
            [_mkv_("Y", TFLOAT, [2, 3, 4])],
            [
                onh.from_array(np.ones(3, dtype=np.float32), name="sc"),
                onh.from_array(np.zeros(3, dtype=np.float32), name="bi"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (2, 3, 4))

    def test_op_lp_normalization(self):
        model = _make_model(
            [oh.make_node("LpNormalization", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [2, 3, 4])],
            [_mkv_("Y", TFLOAT, [2, 3, 4])],
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
        # No axis: input is flattened, output is 1-D with unknown size
        self.assertEqual(b.get_rank("Y"), 1)

    def test_op_compress_axis0(self):
        model = _make_model(
            [oh.make_node("Compress", ["X", "cond"], ["Y"], axis=0)],
            [_mkv_("X", TFLOAT, [3, 4]), _mkv_("cond", TBOOL, [3])],
            [_mkv_("Y", TFLOAT, [None, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        # axis=0: same rank, axis dim is unknown, other dims preserved
        self.assertEqual(b.get_rank("Y"), 2)
        shape = b.get_shape("Y")
        self.assertIsInstance(shape[0], str)
        self.assertEqual(shape[1], 4)

    def test_op_compress_axis1(self):
        model = _make_model(
            [oh.make_node("Compress", ["X", "cond"], ["Y"], axis=1)],
            [_mkv_("X", TFLOAT, [3, 4]), _mkv_("cond", TBOOL, [4])],
            [_mkv_("Y", TFLOAT, [3, None])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        # axis=1: same rank, axis dim is unknown, other dims preserved
        self.assertEqual(b.get_rank("Y"), 2)
        shape = b.get_shape("Y")
        self.assertEqual(shape[0], 3)
        self.assertIsInstance(shape[1], str)

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

    def test_op_conv_auto_pad_same_upper(self):
        model = _make_model(
            [oh.make_node("Conv", ["X", "W"], ["Y"], strides=[2, 2], auto_pad="SAME_UPPER")],
            [_mkv_("X", TFLOAT, [1, 1, 8, 8])],
            [_mkv_("Y", TFLOAT, [1, 1, 4, 4])],
            [onh.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 1, 4, 4))

    def test_op_conv_auto_pad_same_lower(self):
        model = _make_model(
            [oh.make_node("Conv", ["X", "W"], ["Y"], strides=[2, 2], auto_pad="SAME_LOWER")],
            [_mkv_("X", TFLOAT, [1, 1, 8, 8])],
            [_mkv_("Y", TFLOAT, [1, 1, 4, 4])],
            [onh.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 1, 4, 4))

    def test_op_max_pool_ceil_mode(self):
        model = _make_model(
            [
                oh.make_node(
                    "MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1
                )
            ],
            [_mkv_("X", TFLOAT, [1, 1, 7, 7])],
            [_mkv_("Y", TFLOAT, [1, 1, 4, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 1, 4, 4))

    def test_op_max_pool_two_outputs(self):
        model = _make_model(
            [oh.make_node("MaxPool", ["X"], ["Y", "I"], kernel_shape=[2, 2], strides=[2, 2])],
            [_mkv_("X", TFLOAT, [1, 1, 6, 6])],
            [_mkv_("Y", TFLOAT, [1, 1, 3, 3]), _mkv_("I", TINT64, [1, 1, 3, 3])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 1, 3, 3))
        self.assertEqual(b.get_type("I"), TINT64)
        self.assertEqual(b.get_shape("I"), (1, 1, 3, 3))

    def test_op_conv_dynamic_auto_pad(self):
        # Dynamic input dimension with auto_pad triggers symbolic (conv_f3) output
        model = _make_model(
            [oh.make_node("Conv", ["X", "W"], ["Y"], strides=[2, 2], auto_pad="SAME_UPPER")],
            [_mkv_("X", TFLOAT, [1, 1, "N", 8])],
            [_mkv_("Y", TFLOAT, None)],
            [onh.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 1, "conv_f3_0(N,3,2)", 4))

    def test_op_max_pool_kernel_1_dynamic(self):
        # kernel_shape=1 with dynamic spatial dims triggers simplified formula branch
        model = _make_model(
            [oh.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[1, 1])],
            [_mkv_("X", TFLOAT, ["N", 1, "H", 8])],
            [_mkv_("Y", TFLOAT, None)],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), ("N", 1, "H", 8))

    def test_op_max_pool_rank_only(self):
        # Input has rank but no shape: output rank is propagated
        from yobx.xshape.shape_type_compute import set_shape_type_op_any_conv_max_pool

        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 4)
        node = oh.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2])
        set_shape_type_op_any_conv_max_pool(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 4)

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

    def test_op_einsum_matmul(self):
        model = _make_model(
            [oh.make_node("Einsum", ["X", "Y"], ["Z"], equation="ij,jk->ik")],
            [_mkv_("X", TFLOAT, [3, 4]), _mkv_("Y", TFLOAT, [4, 5])],
            [_mkv_("Z", TFLOAT, [3, 5])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (3, 5))

    def test_op_einsum_outer_product(self):
        model = _make_model(
            [oh.make_node("Einsum", ["X", "Y"], ["Z"], equation="i,j->ij")],
            [_mkv_("X", TFLOAT, [3]), _mkv_("Y", TFLOAT, [4])],
            [_mkv_("Z", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (3, 4))

    def test_op_einsum_reduce(self):
        model = _make_model(
            [oh.make_node("Einsum", ["X"], ["Z"], equation="ij->i")],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Z", TFLOAT, [3])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (3,))

    def test_op_einsum_batched_matmul(self):
        model = _make_model(
            [oh.make_node("Einsum", ["X", "Y"], ["Z"], equation="...ij,...jk->...ik")],
            [_mkv_("X", TFLOAT, [2, 3, 4]), _mkv_("Y", TFLOAT, [2, 4, 5])],
            [_mkv_("Z", TFLOAT, [2, 3, 5])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_shape("Z"), (2, 3, 5))

    def test_op_einsum_rank_only(self):
        # When input shapes are unavailable, at least rank should be set.
        b = _TestShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 2)
        b.set_type("Y", TFLOAT)
        b.set_rank("Y", 2)
        node = oh.make_node("Einsum", ["X", "Y"], ["Z"], equation="ij,jk->ik")
        from yobx.xshape.shape_type_compute import set_shape_type_op_any_einsum

        set_shape_type_op_any_einsum(b, node)
        self.assertEqual(b.get_type("Z"), TFLOAT)
        self.assertEqual(b.get_rank("Z"), 2)

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

    def test_op_non_zero_no_rank(self):
        # When input rank is unknown, output is still always 2D
        model = _make_model(
            [oh.make_node("NonZero", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, None)],
            [_mkv_("Y", TINT64, None)],
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TINT64)
        self.assertEqual(b.get_rank("Y"), 2)

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

    def test_op_pad_with_axes(self):
        # Pad only axis 1 with [2, 3] using the optional axes input.
        model = _make_model(
            [oh.make_node("Pad", ["X", "pads", "", "axes"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 9])],
            [
                onh.from_array(np.array([2, 3], dtype=np.int64), name="pads"),
                onh.from_array(np.array([1], dtype=np.int64), name="axes"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 9))

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

    def test_apply_expand_to_shape_no_ones(self):
        # No 1s or -1s in new_shape: return new_shape as-is without entering the loop
        b = _TestShapeBuilder()
        result = b._apply_expand_to_shape((3, 4), (5, 6))
        self.assertEqual(result, (5, 6))

    def test_apply_expand_to_shape_one_copies_input_dim(self):
        # s == 1 in new_shape: copy the corresponding dimension from input_shape
        b = _TestShapeBuilder()
        result = b._apply_expand_to_shape((3, 4), (1, 4))
        self.assertEqual(result, (3, 4))

    def test_apply_expand_to_shape_shorter_input_padded(self):
        # input_shape shorter than new_shape: pad with 1s on the left before expanding
        b = _TestShapeBuilder()
        result = b._apply_expand_to_shape((5,), (3, 1))
        self.assertEqual(result, (3, 5))

    def test_apply_expand_to_shape_zero_dimension(self):
        # s == 0 in new_shape: output 0 at that position (needs 1 elsewhere to enter loop)
        b = _TestShapeBuilder()
        result = b._apply_expand_to_shape((3, 4), (0, 1))
        self.assertEqual(result, (0, 4))

    def test_apply_expand_to_shape_string_dims_equal(self):
        # new_shape dim is a string matching input_shape dim: keep the string
        b = _TestShapeBuilder()
        result = b._apply_expand_to_shape(("batch", 4), ("batch", 1))
        self.assertEqual(result, ("batch", 4))

    def test_apply_expand_to_shape_string_dims_different(self):
        # new_shape dim is a string differing from input_shape dim string: return None
        b = _TestShapeBuilder()
        result = b._apply_expand_to_shape(("batch", 4), ("seq", 1))
        self.assertIsNone(result)

    def test_apply_expand_to_shape_new_str_input_int_one(self):
        # new_shape dim is a string and input dim is 1: use the string from new_shape
        b = _TestShapeBuilder()
        result = b._apply_expand_to_shape((1, 4), ("batch", 1))
        self.assertEqual(result, ("batch", 4))

    def test_apply_expand_to_shape_new_str_input_int_not_one(self):
        # new_shape dim is a string and input dim is an int != 1: return None
        b = _TestShapeBuilder()
        result = b._apply_expand_to_shape((3, 4), ("batch", 1))
        self.assertIsNone(result)

    def test_op_resize_with_sizes(self):
        model = _make_model(
            [oh.make_node("Resize", ["X", "", "", "sizes"], ["Y"])],
            [_mkv_("X", TFLOAT, [1, 3, 4, 5])],
            [_mkv_("Y", TFLOAT, [1, 3, 8, 10])],
            [onh.from_array(np.array([1, 3, 8, 10], dtype=np.int64), name="sizes")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 3, 8, 10))

    def test_op_resize_with_scales(self):
        model = _make_model(
            [oh.make_node("Resize", ["X", "", "scales"], ["Y"])],
            [_mkv_("X", TFLOAT, [1, 3, 4, 5])],
            [_mkv_("Y", TFLOAT, [1, 3, 8, 10])],
            [onh.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 3, 8, 10))

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
        self.assertEqual(b.get_shape("Y"), (3, 6))

    def test_op_slice_no_axes(self):
        # No axes input: default axis order applied
        model = _make_model(
            [oh.make_node("Slice", ["X", "starts", "ends"], ["Y"])],
            [_mkv_("X", TFLOAT, [10, 8])],
            [_mkv_("Y", TFLOAT, [None, None])],
            [
                onh.from_array(np.array([2, 1], dtype=np.int64), name="starts"),
                onh.from_array(np.array([7, 5], dtype=np.int64), name="ends"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_shape("Y"), (5, 4))

    def test_op_slice_with_step(self):
        # step=2 on axis 0
        model = _make_model(
            [oh.make_node("Slice", ["X", "starts", "ends", "axes", "steps"], ["Y"])],
            [_mkv_("X", TFLOAT, [10, 4])],
            [_mkv_("Y", TFLOAT, [None, None])],
            [
                onh.from_array(np.array([0], dtype=np.int64), name="starts"),
                onh.from_array(np.array([10], dtype=np.int64), name="ends"),
                onh.from_array(np.array([0], dtype=np.int64), name="axes"),
                onh.from_array(np.array([2], dtype=np.int64), name="steps"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_shape("Y"), (5, 4))

    def test_op_slice_int64max_end(self):
        # INT64_MAX as end means "slice to the end of the dimension"
        model = _make_model(
            [oh.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])],
            [_mkv_("X", TFLOAT, [5, 6])],
            [_mkv_("Y", TFLOAT, [None, None])],
            [
                onh.from_array(np.array([0], dtype=np.int64), name="starts"),
                onh.from_array(np.array([9223372036854775807], dtype=np.int64), name="ends"),
                onh.from_array(np.array([0], dtype=np.int64), name="axes"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_shape("Y"), (5, 6))

    def test_op_slice_negative_start(self):
        # negative start wraps around: start=-2 on dim=5 → start=3
        model = _make_model(
            [oh.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])],
            [_mkv_("X", TFLOAT, [5, 6])],
            [_mkv_("Y", TFLOAT, [None, None])],
            [
                onh.from_array(np.array([-2], dtype=np.int64), name="starts"),
                onh.from_array(np.array([9223372036854775807], dtype=np.int64), name="ends"),
                onh.from_array(np.array([0], dtype=np.int64), name="axes"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_shape("Y"), (2, 6))

    def test_op_slice_symbolic_dim_preserved(self):
        # When the sliced dimension is symbolic, it cannot be computed; keep as-is
        model = _make_model(
            [oh.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])],
            [_mkv_("X", TFLOAT, ["batch", 6])],
            [_mkv_("Y", TFLOAT, [None, None])],
            [
                onh.from_array(np.array([0], dtype=np.int64), name="starts"),
                onh.from_array(np.array([3], dtype=np.int64), name="ends"),
                onh.from_array(np.array([1], dtype=np.int64), name="axes"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        # axis 1 (integer 6) is sliced: output[1] = 3; axis 0 is "batch" (symbolic)
        self.assertEqual(b.get_shape("Y"), ("batch", 3))

    def test_op_slice_symbolic_dim_gets_newdim(self):
        # When the sliced axis itself is symbolic, a fresh NEWDIM_slice dim is created.
        model = _make_model(
            [oh.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])],
            [_mkv_("X", TFLOAT, ["batch", 6])],
            [_mkv_("Y", TFLOAT, [None, None])],
            [
                onh.from_array(np.array([0], dtype=np.int64), name="starts"),
                onh.from_array(np.array([3], dtype=np.int64), name="ends"),
                onh.from_array(np.array([0], dtype=np.int64), name="axes"),
            ],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        result_shape = b.get_shape("Y")
        # axis 0 ("batch") is sliced → fresh symbolic NEWDIM_slice; axis 1 stays 6
        self.assertEqual(len(result_shape), 2)
        self.assertEqual(result_shape[0], 3)
        self.assertEqual(result_shape[1], 6)

    def test_op_split_value_as_shape(self):
        # splits tensor is NOT a constant but is tracked via value_as_shape
        from yobx.xshape.shape_type_compute import set_shape_type_op_any_split

        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (6, 4))
        # splits tracked as shape value, not a constant
        b.set_value_shape("sp", (3, 3))
        node = oh.make_node("Split", ["X", "sp"], ["A", "B"], axis=0)
        set_shape_type_op_any_split(b, node)
        self.assertEqual(b.get_type("A"), TFLOAT)
        self.assertEqual(b.get_shape("A"), (3, 4))
        self.assertEqual(b.get_shape("B"), (3, 4))

    def test_op_slice_value_as_shape_starts_ends(self):
        # starts/ends are NOT constants but their values are tracked via value_as_shape
        from yobx.xshape.shape_type_compute import set_shape_type_op_any_slice

        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (10, 8))
        # register dynamic starts/ends as shape values (not constants)
        b.set_value_shape("starts", (2,))
        b.set_value_shape("ends", (7,))
        # no axes input: axis 0 is used by default (len(starts)=1)
        node = oh.make_node("Slice", ["X", "starts", "ends"], ["Y"])
        set_shape_type_op_any_slice(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        # axis 0: slice [2:7] on dim 10 → length 5; axis 1 unchanged = 8
        self.assertEqual(b.get_shape("Y"), (5, 8))

    def test_op_slice_value_as_shape_with_dynamic_axes(self):
        # starts/ends/axes are all tracked via value_as_shape (no constants)
        from yobx.xshape.shape_type_compute import set_shape_type_op_any_slice

        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (10, 8))
        b.set_value_shape("starts", (1,))
        b.set_value_shape("ends", (6,))
        b.set_value_shape("axes", (1,))  # slice along axis 1
        node = oh.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])
        set_shape_type_op_any_slice(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        # axis 1: slice [1:6] on dim 8 → length 5; axis 0 unchanged = 10
        self.assertEqual(b.get_shape("Y"), (10, 5))

    def test_op_slice_dynamic_starts_ends_known_axes(self):
        # starts/ends fully dynamic (not constants, not value_as_shape),
        # but axes is a constant → sliced axis gets a fresh dynamic dimension.
        from yobx.xshape.shape_type_compute import set_shape_type_op_any_slice

        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (10, 8))
        # axes is a constant initializer; starts/ends are completely unknown
        b.constants_["axes"] = onh.from_array(np.array([0], dtype=np.int64), name="axes")
        node = oh.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])
        set_shape_type_op_any_slice(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        # axis 1 is unchanged (=8); axis 0 becomes a new dynamic dimension
        result_shape = b.get_shape("Y")
        self.assertEqual(len(result_shape), 2)
        self.assertIsInstance(result_shape[0], str)  # new symbolic dim
        self.assertEqual(result_shape[1], 8)  # unchanged

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
        result = set_shape_type_op_any_sequence_empty(b, node)
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

    def test_op_topk_axis0(self):
        # TopK with axis=0: reduces the first dimension to k.
        model = _make_model(
            [oh.make_node("TopK", ["X", "k"], ["vals", "idx"], axis=0)],
            [_mkv_("X", TFLOAT, [10, 4])],
            [_mkv_("vals", TFLOAT, [3, 4]), _mkv_("idx", TINT64, [3, 4])],
            [onh.from_array(np.array([3], dtype=np.int64), name="k")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_shape("vals"), (3, 4))
        self.assertEqual(b.get_type("vals"), TFLOAT)
        self.assertEqual(b.get_shape("idx"), (3, 4))
        self.assertEqual(b.get_type("idx"), TINT64)

    def test_op_topk_dynamic_k(self):
        # When k is a model input (not a constant), only rank can be inferred.
        model = _make_model(
            [oh.make_node("TopK", ["X", "k"], ["vals", "idx"])],
            [_mkv_("X", TFLOAT, [3, 10]), _mkv_("k", TINT64, [1])],
            [_mkv_("vals", TFLOAT, [3, 5]), _mkv_("idx", TINT64, [3, 5])],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("vals"), TFLOAT)
        self.assertEqual(b.get_type("idx"), TINT64)
        self.assertEqual(b.get_rank("vals"), 2)
        self.assertEqual(b.get_rank("idx"), 2)

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

    def test_op_squeeze_no_axes(self):
        # No axes input and no attribute - all size-1 dimensions are removed.
        model = _make_model(
            [oh.make_node("Squeeze", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [1, 3, 1, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_squeeze_axes_attribute(self):
        # Pre-opset-13 style: axes given as an attribute rather than an input.
        model = _make_model(
            [oh.make_node("Squeeze", ["X"], ["Y"], axes=[0])],
            [_mkv_("X", TFLOAT, [1, 3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
            opset=12,
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_squeeze_negative_axes(self):
        # Negative axis value: axis -2 on shape [3, 1, 4] refers to dim 1.
        model = _make_model(
            [oh.make_node("Squeeze", ["X", "axes"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 1, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
            [onh.from_array(np.array([-2], dtype=np.int64), name="axes")],
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_op_squeeze_rank_only(self):
        # When only rank (not shape) is known the output rank is inferred.
        b = _TestShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 4)
        axes_cst = onh.from_array(np.array([0, 1], dtype=np.int64), name="axes")
        b.set_constant("axes", axes_cst)
        node = oh.make_node("Squeeze", ["X", "axes"], ["Y"])
        set_shape_type_op_any_squeeze(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 2)

    def test_op_squeeze_rank_only_axes_attribute(self):
        # Pre-opset-13 style: axes as attribute, rank-only input.
        b = _TestShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 3)
        node = oh.make_node("Squeeze", ["X"], ["Y"], axes=[0])
        set_shape_type_op_any_squeeze(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 2)

    def test_op_unsqueeze_negative_axes(self):
        # Negative axis: axis -1 on shape [3, 4] inserts dim at the end -> [3, 4, 1].
        model = _make_model(
            [oh.make_node("Unsqueeze", ["X", "axes"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4, 1])],
            [onh.from_array(np.array([-1], dtype=np.int64), name="axes")],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4, 1))

    def test_op_unsqueeze_axes_attribute(self):
        # Pre-opset-13 style: axes given as an attribute rather than an input.
        model = _make_model(
            [oh.make_node("Unsqueeze", ["X"], ["Y"], axes=[0])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [1, 3, 4])],
            opset=12,
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (1, 3, 4))

    def test_op_unsqueeze_rank_only(self):
        # When only rank (not shape) is known the output rank is inferred.
        b = _TestShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 2)
        axes_cst = onh.from_array(np.array([0], dtype=np.int64), name="axes")
        b.set_constant("axes", axes_cst)
        node = oh.make_node("Unsqueeze", ["X", "axes"], ["Y"])
        set_shape_type_op_any_unsqueeze(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 3)

    def test_op_unsqueeze_rank_only_axes_attribute(self):
        # Pre-opset-13 style: axes as attribute, rank-only input.
        b = _TestShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_rank("X", 2)
        node = oh.make_node("Unsqueeze", ["X"], ["Y"], axes=[0])
        set_shape_type_op_any_unsqueeze(b, node)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 3)

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

    def test_op_thresholdedrelu(self):
        model = _make_model(
            [oh.make_node("ThresholdedRelu", ["X"], ["Y"], alpha=1.0)],
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
        b.set_opset("ai.onnx.ml", 3)
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
        b.set_opset("ai.onnx.ml", 5)
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
            "ScatterNDOfShape", ["shape", "idx", "upd"], ["Y"], domain="com.microsoft"
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
            "MultiHeadAttention", ["Q", "K", "V"], ["out1"], domain="com.microsoft"
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
        b.set_opset("ai.onnx.ml", 3)
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

    def test_set_shape_type_custom_local_function(self):
        # Local function: MyUnary(a) -> b where b = Relu(a), same shape as input.
        domain = "local_domain"
        op_type = "MyUnary"
        func_proto = oh.make_function(
            domain,
            op_type,
            inputs=["a"],
            outputs=["b"],
            nodes=[oh.make_node("Relu", ["a"], ["b"])],
            opset_imports=[oh.make_opsetid("", 18)],
        )
        # Pre-populate the function builder with the same input shapes.
        func_builder = _LocalFunctionShapeBuilder()
        func_builder.set_type("a", TFLOAT)
        func_builder.set_shape("a", (3, 4))
        func_builder.set_type("b", TFLOAT)
        func_builder.set_shape("b", (3, 4))
        func_builder._output_names = ["b"]

        # Main builder with the same shapes so re-inference is not triggered.
        b = _LocalFunctionShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.register_local_function(func_proto, func_builder)

        node = oh.make_node(op_type, ["X"], ["Y"], domain=domain)
        set_shape_type_custom(b, node)

        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_set_shape_type_custom_local_function_recompute(self):
        # Same local function but with mismatched shapes to trigger re-inference.
        domain = "local_domain"
        op_type = "MyUnary"
        func_proto = oh.make_function(
            domain,
            op_type,
            inputs=["a"],
            outputs=["b"],
            nodes=[oh.make_node("Relu", ["a"], ["b"])],
            opset_imports=[oh.make_opsetid("", 18)],
        )
        # Function builder has different (stale) input shapes.
        func_builder = _LocalFunctionShapeBuilder()
        func_builder.set_type("a", TFLOAT)
        func_builder.set_shape("a", (5, 6))
        func_builder.set_type("b", TFLOAT)
        func_builder.set_shape("b", (5, 6))
        func_builder._output_names = ["b"]
        func_builder._func_nodes = [oh.make_node("Relu", ["a"], ["b"])]

        # Main builder uses shape (3, 4) — triggers reset + re-inference.
        b = _LocalFunctionShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.register_local_function(func_proto, func_builder)

        node = oh.make_node(op_type, ["X"], ["Y"], domain=domain)
        set_shape_type_custom(b, node)

        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (3, 4))

    def test_supported_ops_in_set_shape_type_custom(self):
        result = supported_ops_in_set_shape_type_custom()
        self.assertIsInstance(result, dict)
        # Check that all three expected domains are present.
        self.assertIn("ai.onnx.ml", result)
        self.assertIn("", result)
        self.assertIn("com.microsoft", result)
        # Check that ai.onnx.ml contains the tree ensemble ops.
        ai_ml_ops = result["ai.onnx.ml"]
        self.assertIn("TreeEnsemble", ai_ml_ops)
        self.assertIn("TreeEnsembleRegressor", ai_ml_ops)
        self.assertIn("TreeEnsembleClassifier", ai_ml_ops)
        # Check that domain-agnostic unary ops are present.
        no_domain_ops = result[""]
        self.assertIn("ReplaceZero", no_domain_ops)
        self.assertIn("NegXplus1", no_domain_ops)
        # Check that custom ops from set_shape_type_op_any_custom are included.
        self.assertIn("FusedMatMul", no_domain_ops)
        self.assertIn("BiasSplitGelu", no_domain_ops)
        # Check that com.microsoft ops are present.
        ms_ops = result["com.microsoft"]
        for op in (
            "Attention",
            "CausalConvWithState",
            "CDist",
            "EmbedLayerNormalization",
            "GatedRelativePositionBias",
            "GreedySearch",
            "GroupQueryAttention",
            "MoE",
            "MurmurHash3",
            "PackedMultiHeadAttention",
            "RelativePositionBias",
        ):
            self.assertIn(op, ms_ops)
        # All values must be frozensets.
        for ops in result.values():
            self.assertIsInstance(ops, frozenset)

    def test_argmax_keepdims(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 4)
        node = oh.make_node("ArgMax", ["X"], ["Y"], axis=1, keepdims=1)
        set_shape_type_op_any_known["ArgMax"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 1, 4))
        self.assertEqual(g._types.get("Y"), TINT64)

    def test_argmin_no_keepdims(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 4)
        node = oh.make_node("ArgMin", ["X"], ["Y"], axis=2, keepdims=0)
        set_shape_type_op_any_known["ArgMin"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 3))
        self.assertEqual(g._types.get("Y"), TINT64)

    def test_global_average_pool(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 8, 4, 4)
        node = oh.make_node("GlobalAveragePool", ["X"], ["Y"])
        set_shape_type_op_any_known["GlobalAveragePool"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 8, 1, 1))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_global_max_pool(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 16, 6, 6)
        node = oh.make_node("GlobalMaxPool", ["X"], ["Y"])
        set_shape_type_op_any_known["GlobalMaxPool"](g, node)
        self.assertEqual(g._shapes.get("Y"), (1, 16, 1, 1))

    def test_flatten_static(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 4)
        node = oh.make_node("Flatten", ["X"], ["Y"], axis=1)
        set_shape_type_op_any_known["Flatten"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 12))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_flatten_dynamic(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = ("batch", 3, 4)
        node = oh.make_node("Flatten", ["X"], ["Y"], axis=1)
        set_shape_type_op_any_known["Flatten"](g, node)
        self.assertEqual(g._shapes.get("Y"), ("batch", 12))

    def test_eyelike_same_type(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (3, 3)
        node = oh.make_node("EyeLike", ["X"], ["Y"])
        set_shape_type_op_any_known["EyeLike"](g, node)
        self.assertEqual(g._shapes.get("Y"), (3, 3))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_eyelike_with_dtype(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (4, 4)
        node = oh.make_node("EyeLike", ["X"], ["Y"], dtype=TINT64)
        set_shape_type_op_any_known["EyeLike"](g, node)
        self.assertEqual(g._shapes.get("Y"), (4, 4))
        self.assertEqual(g._types.get("Y"), TINT64)

    def test_depth_to_space(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 8, 2, 3)
        node = oh.make_node("DepthToSpace", ["X"], ["Y"], blocksize=2)
        set_shape_type_op_any_known["DepthToSpace"](g, node)
        self.assertEqual(g._shapes.get("Y"), (1, 2, 4, 6))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_space_to_depth(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 2, 4, 6)
        node = oh.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)
        set_shape_type_op_any_known["SpaceToDepth"](g, node)
        self.assertEqual(g._shapes.get("Y"), (1, 8, 2, 3))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_gather_elements_axis0(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (3, 4)
        g._types["idx"] = TINT64
        g._shapes["idx"] = (2, 4)
        node = oh.make_node("GatherElements", ["X", "idx"], ["Y"], axis=0)
        set_shape_type_op_any_known["GatherElements"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 4))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_gather_elements_axis1(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (3, 4)
        g._types["idx"] = TINT64
        g._shapes["idx"] = (3, 2)
        node = oh.make_node("GatherElements", ["X", "idx"], ["Y"], axis=1)
        set_shape_type_op_any_known["GatherElements"](g, node)
        self.assertEqual(g._shapes.get("Y"), (3, 2))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_gather_elements_rank_only(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._ranks["X"] = 3
        g._types["idx"] = TINT64
        g._ranks["idx"] = 3
        node = oh.make_node("GatherElements", ["X", "idx"], ["Y"], axis=1)
        set_shape_type_op_any_known["GatherElements"](g, node)
        self.assertEqual(g._types.get("Y"), TFLOAT)
        self.assertEqual(g._ranks.get("Y"), 3)

    # ------------------------------------------------------------------
    # set_shape_type_op_any_gather
    # ------------------------------------------------------------------

    def test_gather_axis0(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (5, 4)
        g._types["idx"] = TINT64
        g._shapes["idx"] = (3,)
        node = oh.make_node("Gather", ["X", "idx"], ["Y"], axis=0)
        set_shape_type_op_any_known["Gather"](g, node)
        self.assertEqual(g._shapes.get("Y"), (3, 4))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_gather_axis1(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (3, 5)
        g._types["idx"] = TINT64
        g._shapes["idx"] = (2,)
        node = oh.make_node("Gather", ["X", "idx"], ["Y"], axis=1)
        set_shape_type_op_any_known["Gather"](g, node)
        self.assertEqual(g._shapes.get("Y"), (3, 2))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_gather_scalar_indices(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (5, 4)
        g._types["idx"] = TINT64
        g._shapes["idx"] = ()
        node = oh.make_node("Gather", ["X", "idx"], ["Y"], axis=0)
        set_shape_type_op_any_known["Gather"](g, node)
        self.assertEqual(g._shapes.get("Y"), (4,))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_gather_nd_general(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 4)
        g._types["idx"] = TINT64
        g._shapes["idx"] = (5, 6)
        node = oh.make_node("Gather", ["X", "idx"], ["Y"], axis=1)
        set_shape_type_op_any_known["Gather"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 5, 6, 4))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_gather_rank_only(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._ranks["X"] = 3
        g._types["idx"] = TINT64
        g._ranks["idx"] = 2
        node = oh.make_node("Gather", ["X", "idx"], ["Y"], axis=0)
        set_shape_type_op_any_known["Gather"](g, node)
        self.assertEqual(g._types.get("Y"), TFLOAT)
        self.assertEqual(g._ranks.get("Y"), 4)

    # ------------------------------------------------------------------
    # set_shape_type_op_any_attention
    # ------------------------------------------------------------------

    def testset_shape_type_op_any_attention_4d(self):
        # 4D input: (batch, head, seq, size)
        g = _MockShapeBuilder()
        for name, shape in [("Q", (2, 8, 10, 64)), ("K", (2, 8, 10, 64)), ("V", (2, 8, 10, 32))]:
            g._types[name] = TFLOAT
            g._shapes[name] = shape
        node = oh.make_node("Attention", ["Q", "K", "V"], ["out"])
        set_shape_type_op_any_attention(g, node)
        self.assertEqual(g._shapes.get("out"), (2, 8, 10, 32))
        self.assertEqual(g._types.get("out"), TFLOAT)

    def testset_shape_type_op_any_attention_3d(self):
        # 3D input: (batch, seq, hidden_size) with q/kv head attributes
        g = _MockShapeBuilder()
        for name, shape in [("Q", (2, 10, 512)), ("K", (2, 10, 256)), ("V", (2, 10, 256))]:
            g._types[name] = TFLOAT
            g._shapes[name] = shape
        node = oh.make_node("Attention", ["Q", "K", "V"], ["out"], q_num_heads=8, kv_num_heads=4)
        set_shape_type_op_any_attention(g, node)
        # v_size = 256 // 4 = 64; output shape = (batch, seq, q_head * v_size) = (2, 10, 512)
        self.assertEqual(g._shapes.get("out"), (2, 10, 512))
        self.assertEqual(g._types.get("out"), TFLOAT)

    def testset_shape_type_op_any_attention_3d_present_outputs(self):
        # 3D with present key/value outputs (output[1] and output[2]), no past inputs
        g = _MockShapeBuilder()
        for name, shape in [("Q", (2, 10, 512)), ("K", (2, 10, 512)), ("V", (2, 10, 512))]:
            g._types[name] = TFLOAT
            g._shapes[name] = shape
        node = oh.make_node(
            "Attention",
            ["Q", "K", "V", "", ""],
            ["out", "present_key", "present_value"],
            q_num_heads=8,
            kv_num_heads=8,
        )
        set_shape_type_op_any_attention(g, node)
        # q_head=8, k_head=v_head=8, k_size=v_size=64
        # output[0]: (2, 10, 8*64) = (2, 10, 512)
        self.assertEqual(g._shapes.get("out"), (2, 10, 512))
        # No past inputs: past_seq=0, total_seq=10+0=10
        # output[1]: (batch, k_head, total_seq, k_size) = (2, 8, 10, 64)
        self.assertEqual(g._shapes.get("present_key"), (2, 8, 10, 64))
        # output[2]: (batch, v_head, total_seq, v_size) = (2, 8, 10, 64)
        self.assertEqual(g._shapes.get("present_value"), (2, 8, 10, 64))

    def testset_shape_type_op_any_attention_rank_only(self):
        # Only rank information available; rank of output[0] = rank of input[0]
        g = _MockShapeBuilder()
        g._types["Q"] = TFLOAT
        g._ranks["Q"] = 3
        node = oh.make_node("Attention", ["Q", "K", "V"], ["out"])
        set_shape_type_op_any_attention(g, node)
        self.assertEqual(g._ranks.get("out"), 3)

    def testset_shape_type_op_any_attention_type_propagation(self):
        # Type from input[0] propagates to all outputs; output[2] uses input[2] type
        g = _MockShapeBuilder()
        g._types["Q"] = TFLOAT16
        g._types["K"] = TFLOAT16
        g._types["V"] = TFLOAT  # different type for v cache output
        node = oh.make_node("Attention", ["Q", "K", "V"], ["out", "present_key", "present_value"])
        set_shape_type_op_any_attention(g, node)
        self.assertEqual(g._types.get("out"), TFLOAT16)
        self.assertEqual(g._types.get("present_key"), TFLOAT16)
        self.assertEqual(g._types.get("present_value"), TFLOAT)

    def testset_shape_type_op_any_softmax_with_shape(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 10)
        node = oh.make_node("Softmax", ["X"], ["Y"])
        set_shape_type_op_any_known["Softmax"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 10))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def testset_shape_type_op_any_softmax_rank_only(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._ranks["X"] = 3
        node = oh.make_node("Softmax", ["X"], ["Y"])
        set_shape_type_op_any_known["Softmax"](g, node)
        self.assertEqual(g._ranks.get("Y"), 3)
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_gridsample_4d_static(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 8, 8)
        g._shapes["grid"] = (2, 5, 6, 2)
        node = oh.make_node("GridSample", ["X", "grid"], ["Y"])
        set_shape_type_op_any_known["GridSample"](g, node)
        self.assertEqual(g._shapes.get("Y"), (2, 3, 5, 6))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_gridsample_5d_static(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 4, 6, 6, 6)
        g._shapes["grid"] = (1, 3, 4, 5, 3)
        node = oh.make_node("GridSample", ["X", "grid"], ["Y"])
        set_shape_type_op_any_known["GridSample"](g, node)
        self.assertEqual(g._shapes.get("Y"), (1, 4, 3, 4, 5))
        self.assertEqual(g._types.get("Y"), TFLOAT)

    def test_gridsample_4d_dynamic(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = ("batch", "channels", "H_in", "W_in")
        g._shapes["grid"] = ("batch", "H_out", "W_out", 2)
        node = oh.make_node("GridSample", ["X", "grid"], ["Y"])
        set_shape_type_op_any_known["GridSample"](g, node)
        self.assertEqual(g._shapes.get("Y"), ("batch", "channels", "H_out", "W_out"))

    def test_gridsample_rank_only(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._ranks["X"] = 4
        node = oh.make_node("GridSample", ["X", "grid"], ["Y"])
        set_shape_type_op_any_known["GridSample"](g, node)
        self.assertEqual(g._ranks.get("Y"), 4)

    def test_blackman_window_known_size(self):
        """BlackmanWindow with constant size sets exact output shape."""
        model = _make_model(
            [oh.make_node("BlackmanWindow", ["size"], ["Y"])],
            [],
            [_mkv_("Y", TFLOAT, [None])],
            [onh.from_array(np.array(16, dtype=np.int64), name="size")],
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (16,))

    def test_blackman_window_with_dtype(self):
        """BlackmanWindow with output_datatype attribute sets correct output type."""
        model = _make_model(
            [oh.make_node("BlackmanWindow", ["size"], ["Y"], output_datatype=TDOUBLE)],
            [],
            [_mkv_("Y", TDOUBLE, [None])],
            [onh.from_array(np.array(8, dtype=np.int64), name="size")],
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TDOUBLE)
        self.assertEqual(b.get_shape("Y"), (8,))

    def test_hamming_window_known_size(self):
        """HammingWindow with constant size sets exact output shape."""
        model = _make_model(
            [oh.make_node("HammingWindow", ["size"], ["Y"])],
            [],
            [_mkv_("Y", TFLOAT, [None])],
            [onh.from_array(np.array(32, dtype=np.int64), name="size")],
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (32,))

    def test_hann_window_known_size(self):
        """HannWindow with constant size sets exact output shape."""
        model = _make_model(
            [oh.make_node("HannWindow", ["size"], ["Y"])],
            [],
            [_mkv_("Y", TFLOAT, [None])],
            [onh.from_array(np.array(64, dtype=np.int64), name="size")],
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_shape("Y"), (64,))

    def test_hann_window_dynamic_size(self):
        """HannWindow with dynamic (non-constant) size sets rank=1."""
        model = _make_model(
            [oh.make_node("HannWindow", ["size"], ["Y"])],
            [_mkv_("size", TINT64, [])],
            [_mkv_("Y", TFLOAT, [None])],
        )
        b = _TestShapeBuilder()
        b.run_model(model)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertEqual(b.get_rank("Y"), 1)


class TestDevicePropagation(ExtTestCase):
    """Tests that device is propagated correctly through each operator."""

    # ------------------------------------------------------------------
    # Einsum
    # ------------------------------------------------------------------

    def test_einsum_device_propagation(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (3, 4)
        g._devices["X"] = -1
        g._types["Y"] = TFLOAT
        g._shapes["Y"] = (4, 5)
        node = oh.make_node("Einsum", ["X", "Y"], ["Z"], equation="ij,jk->ik")
        set_shape_type_op_any_known["Einsum"](g, node)
        self.assertEqual(g._devices.get("Z"), -1)

    # ------------------------------------------------------------------
    # NonZero
    # ------------------------------------------------------------------

    def test_non_zero_device_propagation(self):
        g = _TestShapeBuilder()
        g.set_type("X", TFLOAT)
        g.set_shape("X", (3, 4))
        g.set_device("X", -1)
        node = oh.make_node("NonZero", ["X"], ["Y"])
        set_shape_type_op_any_known["NonZero"](g, node)
        self.assertEqual(g.get_device("Y"), -1)

    # ------------------------------------------------------------------
    # ArgMax / ArgMin
    # ------------------------------------------------------------------

    def test_argmax_device_propagation(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 4)
        g._devices["X"] = -1
        node = oh.make_node("ArgMax", ["X"], ["Y"], axis=1, keepdims=1)
        set_shape_type_op_any_known["ArgMax"](g, node)
        self.assertEqual(g._devices.get("Y"), -1)

    def test_argmin_device_propagation(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 4)
        g._devices["X"] = -1
        node = oh.make_node("ArgMin", ["X"], ["Y"], axis=0, keepdims=0)
        set_shape_type_op_any_known["ArgMin"](g, node)
        self.assertEqual(g._devices.get("Y"), -1)

    # ------------------------------------------------------------------
    # GlobalAveragePool / GlobalMaxPool
    # ------------------------------------------------------------------

    def test_global_average_pool_device_propagation(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 8, 4, 4)
        g._devices["X"] = -1
        node = oh.make_node("GlobalAveragePool", ["X"], ["Y"])
        set_shape_type_op_any_known["GlobalAveragePool"](g, node)
        self.assertEqual(g._devices.get("Y"), -1)

    def test_global_max_pool_device_propagation(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 16, 6, 6)
        g._devices["X"] = -1
        node = oh.make_node("GlobalMaxPool", ["X"], ["Y"])
        set_shape_type_op_any_known["GlobalMaxPool"](g, node)
        self.assertEqual(g._devices.get("Y"), -1)

    # ------------------------------------------------------------------
    # Flatten
    # ------------------------------------------------------------------

    def test_flatten_device_propagation(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (2, 3, 4)
        g._devices["X"] = -1
        node = oh.make_node("Flatten", ["X"], ["Y"], axis=1)
        set_shape_type_op_any_known["Flatten"](g, node)
        self.assertEqual(g._devices.get("Y"), -1)

    # ------------------------------------------------------------------
    # EyeLike
    # ------------------------------------------------------------------

    def test_eyelike_device_propagation(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (3, 3)
        g._devices["X"] = -1
        node = oh.make_node("EyeLike", ["X"], ["Y"])
        set_shape_type_op_any_known["EyeLike"](g, node)
        self.assertEqual(g._devices.get("Y"), -1)

    # ------------------------------------------------------------------
    # DepthToSpace
    # ------------------------------------------------------------------

    def test_depth_to_space_device_propagation(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 8, 2, 3)
        g._devices["X"] = -1
        node = oh.make_node("DepthToSpace", ["X"], ["Y"], blocksize=2)
        set_shape_type_op_any_known["DepthToSpace"](g, node)
        self.assertEqual(g._devices.get("Y"), -1)

    # ------------------------------------------------------------------
    # GridSample
    # ------------------------------------------------------------------

    def test_gridsample_device_propagation(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 3, 4, 4)
        g._devices["X"] = -1
        g._types["grid"] = TFLOAT
        g._shapes["grid"] = (1, 2, 2, 2)
        node = oh.make_node("GridSample", ["X", "grid"], ["Y"])
        set_shape_type_op_any_known["GridSample"](g, node)
        self.assertEqual(g._devices.get("Y"), -1)

    # ------------------------------------------------------------------
    # SpaceToDepth
    # ------------------------------------------------------------------

    def test_space_to_depth_device_propagation(self):
        g = _MockShapeBuilder()
        g._types["X"] = TFLOAT
        g._shapes["X"] = (1, 2, 4, 6)
        g._devices["X"] = -1
        node = oh.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)
        set_shape_type_op_any_known["SpaceToDepth"](g, node)
        self.assertEqual(g._devices.get("Y"), -1)

    # ------------------------------------------------------------------
    # FusedMatMul (transA/transB != 0 case)
    # ------------------------------------------------------------------

    def test_fused_matmul_transA_device_propagation(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (4, 3))
        b.set_device("X", -1)
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (4, 5))
        node = oh.make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft", transA=1)
        set_type_shape_fused_matmul(b, node)
        self.assertEqual(b.get_device("Z"), -1)

    # ------------------------------------------------------------------
    # TreeEnsemble
    # ------------------------------------------------------------------

    def test_tree_ensemble_device_propagation(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (4, 3))
        b.set_device("X", -1)
        b.set_opset("ai.onnx.ml", 3)
        node = oh.make_node(
            "TreeEnsembleRegressor", ["X"], ["Y"], n_targets=2, domain="ai.onnx.ml"
        )
        set_type_shape_tree_ensemble(b, node)
        self.assertEqual(b.get_device("Y"), -1)

    # ------------------------------------------------------------------
    # ToComplex
    # ------------------------------------------------------------------

    def test_to_complex_device_propagation(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4, 2))
        b.set_device("X", -1)
        node = oh.make_node("ToComplex", ["X"], ["Y"], domain="com.microsoft")
        set_type_shape_to_complex(b, node)
        self.assertEqual(b.get_device("Y"), -1)

    # ------------------------------------------------------------------
    # ComplexModule
    # ------------------------------------------------------------------

    def test_complex_module_device_propagation(self):
        b = BasicShapeBuilder()
        b.set_type("X", TCOMPLEX64)
        b.set_shape("X", (3, 4))
        b.set_device("X", -1)
        node = oh.make_node("ComplexModule", ["X"], ["Y"], domain="com.microsoft")
        set_type_shape_complex_module(b, node)
        self.assertEqual(b.get_device("Y"), -1)

    # ------------------------------------------------------------------
    # ScatterNDOfShape
    # ------------------------------------------------------------------

    def test_scatter_nd_of_shape_device_propagation(self):
        b = BasicShapeBuilder()
        b.set_type("updates", TFLOAT)
        b.set_shape("updates", (2, 4))
        b.set_device("updates", -1)
        node = oh.make_node(
            "ScatterNDOfShape", ["shape", "indices", "updates"], ["Y"], domain="com.microsoft"
        )
        set_type_shape_scatter_nd_of_shape(b, node)
        self.assertEqual(b.get_device("Y"), -1)

    # ------------------------------------------------------------------
    # TriMatrix
    # ------------------------------------------------------------------

    def test_tri_matrix_device_propagation(self):
        b = BasicShapeBuilder()
        b.set_type("val", TFLOAT)
        b.set_shape("val", ())
        b.set_device("val", -1)
        node = oh.make_node("TriMatrix", ["shape", "val"], ["Y"], domain="com.microsoft")
        set_type_shape_tri_matrix(b, node)
        self.assertEqual(b.get_device("Y"), -1)

    # ------------------------------------------------------------------
    # Transpose2DCastFP16 / Transpose2DCastFP32
    # ------------------------------------------------------------------

    def test_transpose_2d_cast_fp16_device_propagation(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_device("X", -1)
        node = oh.make_node("Transpose2DCastFP16", ["X"], ["Y"], domain="com.microsoft")
        set_type_shape_transpose_2d_cast_fp16(b, node)
        self.assertEqual(b.get_device("Y"), -1)

    def test_transpose_2d_cast_fp32_device_propagation(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT16)
        b.set_shape("X", (3, 4))
        b.set_device("X", -1)
        node = oh.make_node("Transpose2DCastFP32", ["X"], ["Y"], domain="com.microsoft")
        set_type_shape_transpose_2d_cast_fp32(b, node)
        self.assertEqual(b.get_device("Y"), -1)

    # ------------------------------------------------------------------
    # MultiHeadAttention
    # ------------------------------------------------------------------

    def test_multi_head_attention_device_propagation(self):
        b = BasicShapeBuilder()
        b.set_type("Q", TFLOAT)
        b.set_rank("Q", 3)
        b.set_device("Q", -1)
        node = oh.make_node(
            "MultiHeadAttention", ["Q", "K", "V"], ["out"], domain="com.microsoft"
        )
        set_type_shape_multi_head_attention(b, node)
        self.assertEqual(b.get_device("out"), -1)

    # ------------------------------------------------------------------
    # Reduce (early-return paths)
    # ------------------------------------------------------------------

    def test_reduce_sum_device_propagation_keepdim(self):
        """Device is propagated on the keepdim early-return path."""
        b = _TestShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (2, 3, 4))
        b.set_device("X", -1)
        # Use a model with a non-constant axes input so the keepdim early path is taken
        model = _make_model(
            [oh.make_node("ReduceSum", ["X", "axes"], ["Y"], keepdims=1)],
            [_mkv_("X", TFLOAT, [2, 3, 4]), _mkv_("axes", TINT64, [1])],
            [_mkv_("Y", TFLOAT, [2, 1, 4])],
        )
        b.run_model(model)
        b.set_device("X", -1)
        # Re-run just the node to test device propagation
        from yobx.xshape.shape_type_compute import set_shape_type_op_any_reduce

        node = oh.make_node("ReduceSum", ["X", "axes"], ["Y2"], keepdims=1)
        b.set_type("Y2", TFLOAT)
        set_shape_type_op_any_reduce(b, node)
        self.assertEqual(b.get_device("Y2"), -1)


class TestComputeReshapeShape(ExtTestCase):
    """Unit tests for the compute_reshape_shape helper."""

    # ------------------------------------------------------------------
    # Early-exit: no -1 in shape2
    # ------------------------------------------------------------------

    def test_no_neg1_returns_shape2_unchanged(self):
        # When shape2 has no -1, the function returns shape2 as-is regardless of shape1.
        result = compute_reshape_shape((3, 4), (12,))
        self.assertEqual(result, (12,))

    def test_no_neg1_with_mixed_shapes(self):
        result = compute_reshape_shape((3, 4), (2, 6))
        self.assertEqual(result, (2, 6))

    # ------------------------------------------------------------------
    # All-integer shapes with a single -1
    # ------------------------------------------------------------------

    def test_all_int_clean_division(self):
        # 3*4=12, new shape (2,-1): 12/2=6 → (2, 6)
        result = compute_reshape_shape((3, 4), (2, -1))
        self.assertEqual(result, (2, 6))

    def test_all_int_clean_division_to_1(self):
        # total_int1 == total_int2 (12==12): strict-greater condition fails,
        # so the function falls back to a symbolic "12//12" expression.
        result = compute_reshape_shape((3, 4), (12, -1))
        self.assertEqual((12, 1), result)

    def test_all_int_flatten(self):
        # 2*3*4=24, new shape (-1,): 24 → (24,)
        result = compute_reshape_shape((2, 3, 4), (-1,))
        self.assertEqual(result, (24,))

    def test_all_int_non_divisible_returns_symbolic(self):
        # 3*5=15 is not divisible by 2 → symbolic expression "15//2"
        result = compute_reshape_shape((3, 5), (2, -1))
        self.assertEqual(result[0], 2)
        self.assertIn("15", result[1])
        self.assertIn("2", result[1])

    # ------------------------------------------------------------------
    # Zero dimension in shape1
    # ------------------------------------------------------------------

    def test_zero_in_shape1_yields_zero_for_neg1(self):
        # total_int1 = 0*4 = 0 → -1 becomes 0
        result = compute_reshape_shape((0, 4), (3, -1))
        self.assertEqual(result, (3, 0))

    def test_zero_only_in_shape1(self):
        result = compute_reshape_shape((0,), (-1, 2))
        self.assertEqual(result, (0, 2))

    # ------------------------------------------------------------------
    # Symbolic dimensions
    # ------------------------------------------------------------------

    def test_symbolic_common_dim_with_int_factor(self):
        # shape1=("batch", 4), shape2=("batch", -1)
        # total_int1=4, total_int2=1 (no ints besides -1 in shape2)
        # intpart=4, left1={}, left2={} → ok=4 → ("batch", 4)
        result = compute_reshape_shape(("batch", 4), ("batch", -1))
        self.assertEqual(result, ("batch", 4))

    def test_symbolic_left_in_shape1_only(self):
        # shape1=("batch", "h", 4), shape2=("batch", -1)
        # total_int1=4, total_int2=1, intpart=4
        # left1={"h"}, left2={} → ok="(h)" then "*4" → ("batch", "(h)*4")
        result = compute_reshape_shape(("batch", "h", 4), ("batch", -1))
        self.assertEqual(result[0], "batch")
        self.assertIn("h", result[1])
        self.assertIn("4", result[1])

    def test_symbolic_left_in_shape2_only(self):
        # shape1=(12,), shape2=("d", -1)
        # total_int1=12, total_int2=1, intpart=12
        # left1={}, left2={"d"} → ok="1//(d)" then "*12" → ("d", "1//(d)*12")
        result = compute_reshape_shape((12,), ("d", -1))
        self.assertEqual(result[0], "d")
        self.assertIn("d", result[1])
        self.assertIn("12", result[1])

    def test_symbolic_left_in_both(self):
        # shape1=("a", 4), shape2=("b", -1)
        # total_int1=4, total_int2=1, intpart=4
        # left1={"a"}, left2={"b"} → ok="(a)//((b))"  then "*4"
        result = compute_reshape_shape(("a", 4), ("b", -1))
        self.assertEqual(result[0], "b")
        self.assertIn("a", result[1])
        self.assertIn("b", result[1])


class TestLoopShapeInference(ExtTestCase):
    """Tests for set_shape_type_op_any_loop."""

    def _make_loop_node(self, n_loop_carried, n_scan_outputs, body_shape=(3, 4)):
        """Helper to build a Loop node with a minimal body."""
        v_inputs = [f"v{i}" for i in range(n_loop_carried)]
        v_outs = [f"v_out{i}" for i in range(n_loop_carried)]
        scan_outs = [f"scan{i}" for i in range(n_scan_outputs)]

        body_inputs = [
            oh.make_tensor_value_info("iter", TINT64, []),
            oh.make_tensor_value_info("cond_in", TBOOL, []),
        ] + [oh.make_tensor_value_info(n, TFLOAT, list(body_shape)) for n in v_inputs]

        body_outputs = (
            [oh.make_tensor_value_info("cond_out", TBOOL, [])]
            + [oh.make_tensor_value_info(n, TFLOAT, list(body_shape)) for n in v_outs]
            + [oh.make_tensor_value_info(n, TFLOAT, list(body_shape)) for n in scan_outs]
        )

        # Build minimal body nodes: always use a Constant as a source so the
        # body is valid even when there are no loop-carried variables.
        nodes = []
        if v_inputs:
            nodes.append(oh.make_node("Identity", [v_inputs[0]], [v_outs[0]]))
            for s in scan_outs:
                nodes.append(oh.make_node("Identity", [v_inputs[0]], [s]))
        else:
            # scan-only body: use ConstantOfShape for every scan output
            for s in scan_outs:
                nodes.append(oh.make_node("ConstantOfShape", inputs=["iter"], outputs=[s]))

        body = oh.make_graph(nodes, "body", body_inputs, body_outputs)
        node = oh.make_node(
            "Loop",
            inputs=["max_iter", "cond", *v_inputs],
            outputs=v_outs + [f"stacked_{s}" for s in scan_outs],
            body=body,
        )
        return node

    def test_loop_type_inference_loop_carried_only(self):
        node = self._make_loop_node(1, 0)
        b = _TestShapeBuilder()
        b.set_type("max_iter", TINT64)
        b.set_type("cond", TBOOL)
        b.set_type("v0", TFLOAT)
        b.set_shape("v0", (3, 4))
        b.run_node(node)
        self.assertEqual(b.get_type("v_out0"), TFLOAT)
        self.assertEqual(b.get_shape("v_out0"), (3, 4))

    def test_loop_type_inference_with_scan_output(self):
        node = self._make_loop_node(1, 1)
        b = _TestShapeBuilder()
        b.set_type("max_iter", TINT64)
        b.set_type("cond", TBOOL)
        b.set_type("v0", TFLOAT)
        b.set_shape("v0", (3, 4))
        b.run_node(node)
        # loop-carried output
        self.assertEqual(b.get_type("v_out0"), TFLOAT)
        self.assertEqual(b.get_shape("v_out0"), (3, 4))
        # scan output has an extra leading dimension
        self.assertEqual(b.get_type("stacked_scan0"), TFLOAT)
        scan_shape = b.get_shape("stacked_scan0")
        self.assertEqual(len(scan_shape), 3)  # (iters, 3, 4)
        self.assertIsInstance(scan_shape[0], str)  # symbolic iter dim
        self.assertEqual(scan_shape[1:], (3, 4))

    def test_loop_type_inference_no_body_returns_none(self):
        node = oh.make_node("Loop", inputs=["max_iter", "cond", "v0"], outputs=["v_final"])
        b = _MockShapeBuilder()
        result = set_shape_type_op_any_loop(b, node)
        self.assertIsNone(result)

    def test_loop_type_inferred_from_body_graph_when_missing(self):
        """When body output elem_types are undeclared (0), they should be
        inferred by propagating types through the body graph nodes."""
        body = oh.make_graph(
            [
                oh.make_node("Add", ["v", "v"], ["v_out"]),
                oh.make_node("Identity", ["v"], ["scan_out"]),
            ],
            "loop_body",
            [
                oh.make_tensor_value_info("iter", TINT64, []),
                oh.make_tensor_value_info("cond_in", TBOOL, []),
                # input type declared as FLOAT
                oh.make_tensor_value_info("v", TFLOAT, [3, 4]),
            ],
            [
                oh.make_tensor_value_info("cond_out", TBOOL, []),
                # output types intentionally undeclared (0)
                oh.make_tensor_value_info("v_out", 0, None),
                oh.make_tensor_value_info("scan_out", 0, None),
            ],
        )
        node = oh.make_node(
            "Loop", inputs=["max_iter", "cond", "v_in"], outputs=["v_final", "scan"], body=body
        )
        b = _TestShapeBuilder()
        b.set_type("max_iter", TINT64)
        b.set_type("cond", TBOOL)
        b.set_type("v_in", TFLOAT)
        b.set_shape("v_in", (3, 4))
        b.run_node(node)
        # types should be inferred even though undeclared in the body outputs
        self.assertEqual(b.get_type("v_final"), TFLOAT)
        self.assertEqual(b.get_type("scan"), TFLOAT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
