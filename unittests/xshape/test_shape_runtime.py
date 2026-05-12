import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape import BasicShapeBuilder

TINT64 = onnx.TensorProto.INT64
TFLOAT = onnx.TensorProto.FLOAT


class _TestShapeBuilder(BasicShapeBuilder):
    """BasicShapeBuilder extended with helpers needed by _ShapeRuntime."""

    as_function = False
    _debug_quiet = False

    def unique_dimension_name(self, prefix: str) -> str:
        return f"{prefix}_0"

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


def _int64_cst(name, data):
    """Create a named int64 TensorProto constant."""
    return onh.from_array(np.array(data, dtype=np.int64), name=name)


class TestShapeRuntime(ExtTestCase):
    # ------------------------------------------------------------------
    # simple_update_value_shape_with_node - dispatch / guard tests
    # ------------------------------------------------------------------

    def test_simple_update_unknown_domain_returns_false(self):
        b = _TestShapeBuilder()
        node = oh.make_node("Add", ["x", "y"], ["z"], domain="custom")
        self.assertFalse(b.simple_update_value_shape_with_node(node))

    def test_simple_update_unknown_op_returns_false(self):
        b = _TestShapeBuilder()
        node = oh.make_node("SomeUnknownOp", ["x"], ["y"])
        self.assertFalse(b.simple_update_value_shape_with_node(node))

    def test_simple_update_auto_extracts_int64_constant(self):
        # An INT64 1-D constant input should be auto-promoted to a value shape.
        b = _TestShapeBuilder()
        b._known_value_shape["x"] = (2, 3, 4)
        b.set_shape("x", (3,))
        b.set_type("x", TINT64)
        b.set_constant("idx", _int64_cst("idx", [1]))
        node = oh.make_node("Gather", ["x", "idx"], ["out"])
        result = b.simple_update_value_shape_with_node(node)
        self.assertTrue(result)
        # Gather(x, idx=[1]) on shape (2,3,4) → element at index 1 → (3,)
        self.assertEqual(b.value_as_shape("out"), (3,))

    # ------------------------------------------------------------------
    # _update_value_shape_with_node_Identity
    # ------------------------------------------------------------------

    def test_identity_propagates_value_shape(self):
        b = _TestShapeBuilder()
        b._known_value_shape["x"] = (2, 3, 4)
        b.set_shape("x", (3,))
        b.set_type("x", TINT64)
        node = oh.make_node("Identity", ["x"], ["y"])
        result = b._update_value_shape_with_node_Identity(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("y"), (2, 3, 4))

    def test_identity_no_value_shape_returns_false(self):
        b = _TestShapeBuilder()
        b.set_shape("x", (3,))
        b.set_type("x", TFLOAT)
        node = oh.make_node("Identity", ["x"], ["y"])
        result = b._update_value_shape_with_node_Identity(node)
        self.assertFalse(result)

    def test_identity_propagates_string_value_shape(self):
        b = _TestShapeBuilder()
        b._known_value_shape["x"] = "batch"
        b.set_shape("x", ())
        b.set_type("x", TINT64)
        node = oh.make_node("Identity", ["x"], ["y"])
        result = b._update_value_shape_with_node_Identity(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("y"), "batch")

    # ------------------------------------------------------------------
    # _update_value_shape_with_node_Abs
    # ------------------------------------------------------------------

    def test_abs_propagates_string_value_shape(self):
        # For non-numeric value shapes, Abs just propagates like Identity.
        b = _TestShapeBuilder()
        b._known_value_shape["x"] = "batch"
        b.set_shape("x", ())
        b.set_type("x", TINT64)
        node = oh.make_node("Abs", ["x"], ["y"])
        result = b._update_value_shape_with_node_Abs(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("y"), "batch")

    # ------------------------------------------------------------------
    # _update_value_shape_with_node_Shape
    # ------------------------------------------------------------------

    def test_shape_no_attrs_known_shape(self):
        b = _TestShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (2, 3, 4))
        node = oh.make_node("Shape", ["X"], ["Y"])
        result = b._update_value_shape_with_node_Shape(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("Y"), (2, 3, 4))
        self.assertEqual(b.get_shape("Y"), (3,))

    def test_shape_no_attrs_unknown_shape_returns_false(self):
        b = _TestShapeBuilder()
        b.set_rank("X", 3)
        node = oh.make_node("Shape", ["X"], ["Y"])
        result = b._update_value_shape_with_node_Shape(node)
        self.assertFalse(result)

    def test_shape_with_start_attr(self):
        b = _TestShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (2, 3, 4, 5))
        node = oh.make_node("Shape", ["X"], ["Y"], start=2)
        result = b._update_value_shape_with_node_Shape(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("Y"), (4, 5))

    def test_shape_with_start_end_attrs(self):
        b = _TestShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", (2, 3, 4, 5))
        node = oh.make_node("Shape", ["X"], ["Y"], start=1, end=3)
        result = b._update_value_shape_with_node_Shape(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("Y"), (3, 4))

    def test_shape_with_start_attr_end_missing_unknown_shape(self):
        # start is set, end is missing, input has no shape or rank known:
        # the function should return False and store a string value shape.
        b = _TestShapeBuilder()
        node = oh.make_node("Shape", ["X"], ["Y"], start=2)
        result = b._update_value_shape_with_node_Shape(node)
        self.assertFalse(result)
        self.assertEqual(b.value_as_shape("Y"), "X[2:]")

    # ------------------------------------------------------------------
    # _update_value_shape_with_node_Gather
    # ------------------------------------------------------------------

    def test_gather_tuple_ndarray_scalar_index(self):
        b = _TestShapeBuilder()
        b._known_value_shape["data"] = (10, 20, 30)
        b.set_shape("data", (3,))
        b.set_type("data", TINT64)
        b.set_constant("idx", _int64_cst("idx", 2))
        node = oh.make_node("Gather", ["data", "idx"], ["out"])
        result = b._update_value_shape_with_node_Gather(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), 30)

    def test_gather_str_int_index(self):
        # Gather on a string shape with a Python-int constant.
        b = _TestShapeBuilder()
        b._known_value_shape["data"] = "shapeX"
        b.set_shape("data", ())
        b.set_type("data", TINT64)
        # Store a Python int as the pre-computed constant value.
        b.constants_["idx"] = _int64_cst("idx", 0)
        b.constants_computed_["idx"] = 0  # Python int
        b.set_shape("idx", ())
        b.set_type("idx", TINT64)
        node = oh.make_node("Gather", ["data", "idx"], ["out"])
        result = b._update_value_shape_with_node_Gather(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), "shapeX[0]")

    def test_gather_tuple_int_index(self):
        # Gather on a tuple shape with a Python-int constant → element at index.
        b = _TestShapeBuilder()
        b._known_value_shape["data"] = (5, 10, 15)
        b.set_shape("data", (3,))
        b.set_type("data", TINT64)
        b.constants_["idx"] = _int64_cst("idx", 1)
        b.constants_computed_["idx"] = 1  # Python int
        b.set_shape("idx", ())
        b.set_type("idx", TINT64)
        node = oh.make_node("Gather", ["data", "idx"], ["out"])
        result = b._update_value_shape_with_node_Gather(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), 10)

    def test_gather_no_constant_returns_false(self):
        b = _TestShapeBuilder()
        b._known_value_shape["data"] = (10, 20, 30)
        b.set_shape("data", (3,))
        b.set_type("data", TINT64)
        b.set_shape("idx", ())
        b.set_type("idx", TINT64)
        node = oh.make_node("Gather", ["data", "idx"], ["out"])
        result = b._update_value_shape_with_node_Gather(node)
        self.assertFalse(result)

    # ------------------------------------------------------------------
    # _update_value_shape_with_node_Squeeze
    # ------------------------------------------------------------------

    def test_squeeze_tuple_axis0(self):
        b = _TestShapeBuilder()
        b._known_value_shape["x"] = (42,)
        b.set_shape("x", (1,))
        b.set_type("x", TINT64)
        b.set_constant("axes", _int64_cst("axes", 0))
        node = oh.make_node("Squeeze", ["x", "axes"], ["y"])
        result = b._update_value_shape_with_node_Squeeze(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("y"), 42)

    def test_squeeze_str_axis0(self):
        # Squeeze a string shape value along axis 0 → "squeeze(s)".
        b = _TestShapeBuilder()
        b._known_value_shape["x"] = "someShape"
        b.set_shape("x", ())
        b.set_type("x", TINT64)
        b.set_constant("axes", _int64_cst("axes", 0))
        node = oh.make_node("Squeeze", ["x", "axes"], ["y"])
        result = b._update_value_shape_with_node_Squeeze(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("y"), "squeeze(someShape)")

    # ------------------------------------------------------------------
    # _update_value_shape_with_values_Concat
    # ------------------------------------------------------------------

    def test_concat_two_shape_tuples(self):
        b = _TestShapeBuilder()
        b._known_value_shape["a"] = (2, 3)
        b._known_value_shape["b"] = (4, 5)
        b.set_shape("a", (2,))
        b.set_type("a", TINT64)
        b.set_shape("b", (2,))
        b.set_type("b", TINT64)
        node = oh.make_node("Concat", ["a", "b"], ["out"], axis=0)
        values = [(2, 3), (4, 5)]
        result = b._update_value_shape_with_values_Concat(node, values)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (2, 3, 4, 5))

    def test_concat_scalar_and_tuple(self):
        b = _TestShapeBuilder()
        b.set_shape("out", (3,))
        b.set_type("out", TINT64)
        node = oh.make_node("Concat", ["a", "b"], ["out"], axis=0)
        values = [1, (2, 3)]
        result = b._update_value_shape_with_values_Concat(node, values)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (1, 2, 3))

    # ------------------------------------------------------------------
    # _update_value_shape_with_values_Range
    # ------------------------------------------------------------------

    def test_range_from_int_values(self):
        b = _TestShapeBuilder()
        b.set_shape("out", (5,))
        b.set_type("out", TINT64)
        node = oh.make_node("Range", ["start", "end", "step"], ["out"])
        values = [0, 5, 1]
        result = b._update_value_shape_with_values_Range(node, values)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (0, 1, 2, 3, 4))

    def test_range_with_non_one_step(self):
        b = _TestShapeBuilder()
        b.set_shape("out", (3,))
        b.set_type("out", TINT64)
        node = oh.make_node("Range", ["start", "end", "step"], ["out"])
        values = [0, 6, 2]
        result = b._update_value_shape_with_values_Range(node, values)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (0, 2, 4))

    # ------------------------------------------------------------------
    # _update_value_shape_with_values_element_wise (Add/Sub/Mul/Div/Mod)
    # ------------------------------------------------------------------

    def test_add_two_ints(self):
        b = _TestShapeBuilder()
        b.set_shape("out", ())
        b.set_type("out", TINT64)
        node = oh.make_node("Add", ["a", "b"], ["out"])
        result = b._update_value_shape_with_values_Add(node, [3, 4])
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), 7)

    def test_sub_two_ints(self):
        b = _TestShapeBuilder()
        b.set_shape("out", ())
        b.set_type("out", TINT64)
        node = oh.make_node("Sub", ["a", "b"], ["out"])
        result = b._update_value_shape_with_values_Sub(node, [10, 3])
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), 7)

    def test_mul_two_ints(self):
        b = _TestShapeBuilder()
        b.set_shape("out", ())
        b.set_type("out", TINT64)
        node = oh.make_node("Mul", ["a", "b"], ["out"])
        result = b._update_value_shape_with_values_Mul(node, [3, 4])
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), 12)

    def test_div_two_ints(self):
        b = _TestShapeBuilder()
        b.set_shape("out", ())
        b.set_type("out", TINT64)
        node = oh.make_node("Div", ["a", "b"], ["out"])
        result = b._update_value_shape_with_values_Div(node, [12, 4])
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), 3)

    def test_mod_two_ints(self):
        b = _TestShapeBuilder()
        b.set_shape("out", ())
        b.set_type("out", TINT64)
        node = oh.make_node("Mod", ["a", "b"], ["out"])
        result = b._update_value_shape_with_values_Mod(node, [10, 3])
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), 1)

    def test_add_str_and_int(self):
        b = _TestShapeBuilder()
        b.set_shape("out", ())
        b.set_type("out", TINT64)
        node = oh.make_node("Add", ["a", "b"], ["out"])
        result = b._update_value_shape_with_values_Add(node, ["batch", 1])
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), "batch+1")

    def test_add_tuples_element_wise(self):
        b = _TestShapeBuilder()
        b.set_shape("out", (2,))
        b.set_type("out", TINT64)
        node = oh.make_node("Add", ["a", "b"], ["out"])
        result = b._update_value_shape_with_values_Add(node, [(2, 3), (1, 4)])
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (3, 7))

    def test_add_broadcast_scalar_to_tuple(self):
        b = _TestShapeBuilder()
        b.set_shape("out", (3,))
        b.set_type("out", TINT64)
        node = oh.make_node("Add", ["a", "b"], ["out"])
        result = b._update_value_shape_with_values_Add(node, [1, (2, 3, 4)])
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (3, 4, 5))

    # ------------------------------------------------------------------
    # _update_value_shape_with_values_Gather
    # ------------------------------------------------------------------

    def test_gather_values_tuple_indices(self):
        b = _TestShapeBuilder()
        b.set_shape("out", (2,))
        b.set_type("out", TINT64)
        node = oh.make_node("Gather", ["data", "idx"], ["out"])
        values = [(10, 20, 30, 40), (0, 2)]
        result = b._update_value_shape_with_values_Gather(node, values)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (10, 30))

    # ------------------------------------------------------------------
    # _update_value_shape_with_values_Slice
    # ------------------------------------------------------------------

    def test_slice_identity_full_range(self):
        # starts=(0,), ends=(INT64_MAX,) → identity slice
        b = _TestShapeBuilder()
        b.set_shape("out", (3,))
        b.set_type("out", TINT64)
        node = oh.make_node("Slice", ["data", "starts", "ends"], ["out"])
        values = [(5, 10, 15), (0,), (9223372036854775807,)]
        result = b._update_value_shape_with_values_Slice(node, values)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (5, 10, 15))

    def test_slice_with_axes(self):
        # axes=(0,), starts=(1,), ends=(3,) → data[1:3]
        b = _TestShapeBuilder()
        b.set_shape("out", (2,))
        b.set_type("out", TINT64)
        node = oh.make_node("Slice", ["data", "starts", "ends", "axes"], ["out"])
        values = [(5, 10, 15, 20), (1,), (3,), (0,)]
        result = b._update_value_shape_with_values_Slice(node, values)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (10, 15))

    def test_slice_no_axes_non_full_range_returns_false(self):
        # len(values)==3 but not a full-range identity → #SV-Sl/2
        b = _TestShapeBuilder()
        b.set_shape("out", (2,))
        b.set_type("out", TINT64)
        node = oh.make_node("Slice", ["data", "starts", "ends"], ["out"])
        values = [(5, 10, 15), (1,), (3,)]
        result = b._update_value_shape_with_values_Slice(node, values)
        self.assertFalse(result)

    def test_slice_non_zero_axis_returns_false(self):
        # axes=(1,) (not axis 0) → #SV-Sl/2
        b = _TestShapeBuilder()
        b.set_shape("out", (2,))
        b.set_type("out", TINT64)
        node = oh.make_node("Slice", ["data", "starts", "ends", "axes"], ["out"])
        values = [(5, 10, 15, 20), (1,), (3,), (1,)]
        result = b._update_value_shape_with_values_Slice(node, values)
        self.assertFalse(result)

    def test_slice_dynamic_end_returns_false(self):
        # axes=(0,), starts=(0,), ends=("dim",) (symbolic) → #SV-Sl/3
        b = _TestShapeBuilder()
        b.set_shape("out", (3,))
        b.set_type("out", TINT64)
        node = oh.make_node("Slice", ["data", "starts", "ends", "axes"], ["out"])
        values = [("a", "b", "c"), (0,), ("dim",), (0,)]
        result = b._update_value_shape_with_values_Slice(node, values)
        self.assertFalse(result)

    def test_slice_with_step(self):
        # 5 inputs with axes=(0,), starts=(1,), ends=(5,), step=(2,) → #SV-Sl4
        b = _TestShapeBuilder()
        b.set_shape("out", (2,))
        b.set_type("out", TINT64)
        node = oh.make_node("Slice", ["data", "starts", "ends", "axes", "steps"], ["out"])
        values = [(5, 10, 15, 20, 25), (1,), (5,), (0,), (2,)]
        result = b._update_value_shape_with_values_Slice(node, values)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (10, 20))

    # ------------------------------------------------------------------
    # _update_value_shape_with_node_Unsqueeze
    # ------------------------------------------------------------------

    def test_unsqueeze_multi_element_tuple_returns_false(self):
        # values_0 is tuple with len > 1 → #SV-Unsq/1
        b = _TestShapeBuilder()
        b._known_value_shape["x"] = (2, 3)
        b.set_shape("x", (2,))
        b.set_type("x", TINT64)
        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        result = b._update_value_shape_with_node_Unsqueeze(node)
        self.assertFalse(result)

    def test_unsqueeze_rank_gt_zero_returns_false(self):
        # rank > 0 and values_0 is None → #SV-Unsq/2
        b = _TestShapeBuilder()
        b.set_rank("x", 1)
        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        result = b._update_value_shape_with_node_Unsqueeze(node)
        self.assertFalse(result)

    def test_unsqueeze_no_rank_no_value_shape_returns_false(self):
        # no rank and no value_shape → #SV-Unsq/3
        b = _TestShapeBuilder()
        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        result = b._update_value_shape_with_node_Unsqueeze(node)
        self.assertFalse(result)

    def test_unsqueeze_scalar_axes_input_no_value_shape(self):
        # rank=0, axes input=[0], values_0=None → #SV-Unsq4, output=("x",)
        b = _TestShapeBuilder()
        b.set_rank("x", 0)
        b.set_type("x", onnx.TensorProto.INT64)
        b.set_constant("axes", _int64_cst("axes", [0]))
        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        result = b._update_value_shape_with_node_Unsqueeze(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("y"), ("x",))

    def test_unsqueeze_scalar_axes_input_with_value_shape(self):
        # rank=0, axes input=[0], values_0=42 → #SV-Unsq4, output=(42,)
        b = _TestShapeBuilder()
        b.set_rank("x", 0)
        b._known_value_shape["x"] = 42
        b.set_constant("axes", _int64_cst("axes", [0]))
        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        result = b._update_value_shape_with_node_Unsqueeze(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("y"), (42,))

    def test_unsqueeze_scalar_0d_axes_returns_false(self):
        # rank=0, axes input is 0-D scalar → cst=tuple(), len≠1 → return False
        b = _TestShapeBuilder()
        b.set_rank("x", 0)
        b.set_constant("axes", _int64_cst("axes", 0))
        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        result = b._update_value_shape_with_node_Unsqueeze(node)
        self.assertFalse(result)

    # ------------------------------------------------------------------
    # Gather with multi-element ndarray index (#SV-Ga8)
    # ------------------------------------------------------------------

    def test_gather_tuple_ndarray_multi_element_index(self):
        # Gather on a concrete shape tuple with a two-element int64 index array.
        # Gather((2, 3, 8, 4), [0, 1]) must yield value_shape (2, 3) and shape (2,).
        # This covers the case that arises for multi-batch 4-D einsum equations.
        b = _TestShapeBuilder()
        b._known_value_shape["data"] = (2, 3, 8, 4)
        b.set_shape("data", (4,))
        b.set_type("data", TINT64)
        b.set_constant("idx", _int64_cst("idx", [0, 1]))
        node = oh.make_node("Gather", ["data", "idx"], ["out"])
        result = b._update_value_shape_with_node_Gather(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (2, 3))
        self.assertEqual(b.get_shape("out"), (2,))

    def test_gather_symbolic_tuple_ndarray_multi_element_index(self):
        # Gather on a symbolic shape tuple with a two-element int64 index array.
        # Gather(('A', 'B', 'I', 'K'), [0, 1]) must yield value_shape ('A', 'B')
        # and output shape (2,).
        b = _TestShapeBuilder()
        b._known_value_shape["data"] = ("A", "B", "I", "K")
        b.set_shape("data", (4,))
        b.set_type("data", TINT64)
        b.set_constant("idx", _int64_cst("idx", [0, 1]))
        node = oh.make_node("Gather", ["data", "idx"], ["out"])
        result = b._update_value_shape_with_node_Gather(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), ("A", "B"))
        self.assertEqual(b.get_shape("out"), (2,))

    def test_gather_tuple_ndarray_three_element_index(self):
        # Gather on a concrete shape tuple with a three-element int64 index array.
        # Gather((2, 3, 8, 4), [0, 1, 2]) must yield value_shape (2, 3, 8)
        # and output shape (3,).
        b = _TestShapeBuilder()
        b._known_value_shape["data"] = (2, 3, 8, 4)
        b.set_shape("data", (4,))
        b.set_type("data", TINT64)
        b.set_constant("idx", _int64_cst("idx", [0, 1, 2]))
        node = oh.make_node("Gather", ["data", "idx"], ["out"])
        result = b._update_value_shape_with_node_Gather(node)
        self.assertTrue(result)
        self.assertEqual(b.value_as_shape("out"), (2, 3, 8))
        self.assertEqual(b.get_shape("out"), (3,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
