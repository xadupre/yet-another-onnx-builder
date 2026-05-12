import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape import BasicShapeBuilder

TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64
_mkv_ = oh.make_tensor_value_info


class TestMakeDimensionNameIfNecessary(ExtTestCase):
    def setUp(self):
        self.b = BasicShapeBuilder()

    def test_same_values_caret(self):
        result = self.b.make_dimension_name_if_necessary("A", "A", "^")
        self.assertEqual(result, "A")

    def test_a_ends_with_caret_b(self):
        result = self.b.make_dimension_name_if_necessary("x^2", "2", "^")
        self.assertEqual(result, "x^2")

    def test_b_starts_with_a_caret(self):
        result = self.b.make_dimension_name_if_necessary("x", "x^2", "^")
        self.assertEqual(result, "x^2")

    def test_add_integers(self):
        result = self.b.make_dimension_name_if_necessary(2, 3, "+")
        self.assertEqual(result, "2+3")

    def test_add_strings(self):
        result = self.b.make_dimension_name_if_necessary("a", "b", "+")
        self.assertEqual(result, "a+b")

    def test_complex_expression_wrapped(self):
        # 'a+b' contains '+' so should be wrapped
        result = self.b.make_dimension_name_if_necessary("a+b", "c", "*")
        self.assertEqual(result, "(a+b)*c")

    def test_complex_b_wrapped(self):
        result = self.b.make_dimension_name_if_necessary("a", "b+c", "*")
        self.assertEqual(result, "a*(b+c)")

    def test_both_complex_wrapped(self):
        result = self.b.make_dimension_name_if_necessary("a+b", "c*d", "^")
        self.assertEqual(result, "(a+b)^(c*d)")

    def test_caret_different_no_prefix(self):
        result = self.b.make_dimension_name_if_necessary("x", "y", "^")
        self.assertEqual(result, "x^y")


class TestUpdateNodeConstant(ExtTestCase):
    def setUp(self):
        self.b = BasicShapeBuilder()

    def test_update_node_constant_with_none(self):
        # node=None means it's an initializer constant
        result = self.b.update_node_constant("c", None)
        self.assertTrue(result)
        self.assertIn("c", self.b.constants_)

    def test_update_node_constant_with_node(self):
        node = oh.make_node("Identity", ["x"], ["y"])
        self.b.constants_["x"] = None
        result = self.b.update_node_constant("y", node)
        self.assertTrue(result)
        self.assertIn("y", self.b.constants_)

    def test_update_node_constant_random_skipped(self):
        node = oh.make_node("RandomNormal", [], ["r"])
        result = self.b.update_node_constant("r", node)
        self.assertFalse(result)
        self.assertNotIn("r", self.b.constants_)

    def test_update_node_constant_invalid_name_type(self):
        self.assertRaises(AssertionError, self.b.update_node_constant, 42, None)

    def test_update_node_constant_with_constant_node(self):
        value = oh.make_tensor("value", TFLOAT, [2], [1.0, 2.0])
        node = oh.make_node("Constant", [], ["c"], value=value)
        result = self.b.update_node_constant("c", node)
        self.assertTrue(result)
        self.assertIn("c", self.b.constants_)

    def test_update_node_constant_invalid_node_type(self):
        self.assertRaises(AssertionError, self.b.update_node_constant, "c", "not_a_node")


class TestMakeNodeSetTypeShapeConstantIdentity(ExtTestCase):
    def test_identity_propagates_shape_and_type(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3, 4))
        node = oh.make_node("Identity", ["x"], ["y"])
        b.constants_["x"] = None
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("y"), TFLOAT)
        self.assertEqual(b.get_shape("y"), (3, 4))

    def test_identity_propagates_rank_only(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_rank("x", 2)
        node = oh.make_node("Identity", ["x"], ["y"])
        b.constants_["x"] = None
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_rank("y"), 2)

    def test_identity_propagates_device(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3,))
        b.set_device("x", -1)
        node = oh.make_node("Identity", ["x"], ["y"])
        b.constants_["x"] = None
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_device("y"), -1)

    def test_identity_marks_constant(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3,))
        b.constants_["x"] = None
        node = oh.make_node("Identity", ["x"], ["y"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertTrue(b.is_constant("y"))


class TestMakeNodeSetTypeShapeConstantShape(ExtTestCase):
    def test_shape_sets_type_and_shape(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3, 4, 5))
        node = oh.make_node("Shape", ["x"], ["s"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("s"), TINT64)
        self.assertEqual(b.get_shape("s"), (3,))

    def test_shape_with_start_end(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3, 4, 5))
        node = oh.make_node("Shape", ["x"], ["s"], start=1, end=3)
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_shape("s"), (2,))

    def test_shape_no_rank(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        node = oh.make_node("Shape", ["x"], ["s"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("s"), TINT64)
        self.assertEqual(b.get_rank("s"), 1)

    def test_shape_marks_constant_when_static(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3, 4, 5))
        node = oh.make_node("Shape", ["x"], ["s"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertTrue(b.is_constant("s"))


class TestMakeNodeSetTypeShapeConstantSize(ExtTestCase):
    def test_size_sets_type_and_shape(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3, 4))
        node = oh.make_node("Size", ["x"], ["sz"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("sz"), TINT64)
        self.assertEqual(b.get_shape("sz"), ())

    def test_size_marks_constant(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3, 4))
        b.constants_["x"] = None
        node = oh.make_node("Size", ["x"], ["sz"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertTrue(b.is_constant("sz"))


class TestMakeNodeSetTypeShapeConstantReshape(ExtTestCase):
    def test_reshape_static_shape(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (6, 4))
        cst = onh.from_array(np.array([3, 8], dtype=np.int64), name="sh")
        b.set_constant("sh", cst)
        node = oh.make_node("Reshape", ["x", "sh"], ["y"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("y"), TFLOAT)
        self.assertEqual(b.get_shape("y"), (3, 8))

    def test_reshape_minus_one(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (6, 4))
        cst = onh.from_array(np.array([3, -1], dtype=np.int64), name="sh")
        b.set_constant("sh", cst)
        node = oh.make_node("Reshape", ["x", "sh"], ["y"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("y"), TFLOAT)
        self.assertEqual(b.get_shape("y"), (3, 8))

    def test_reshape_with_zero(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3, 4, 5))
        cst = onh.from_array(np.array([0, 4, 5], dtype=np.int64), name="sh")
        b.set_constant("sh", cst)
        node = oh.make_node("Reshape", ["x", "sh"], ["y"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("y"), TFLOAT)
        self.assertEqual(b.get_shape("y"), (3, 4, 5))

    def test_reshape_with_zero_rank_increase(self):
        # Reshape (0, 0) → (1, 0, 0): zeros in the target shape are at positions
        # 1 and 2, but the input is rank 2 with dynamic dims (stored as 0).
        # The output rank (3) > input rank (2), so copy-from-input semantics
        # cannot be satisfied for position 2 — shape inference should gracefully
        # skip rather than raising AssertionError.
        b = BasicShapeBuilder()
        b.set_type("X1", TFLOAT)
        # Dynamic 2D input — dimensions stored as 0 (unknown).
        b.set_shape("X1", (0, 0))
        cst = onh.from_array(np.array([1, 0, 0], dtype=np.int64), name="sh")
        b.set_constant("sh", cst)
        node = oh.make_node("Reshape", ["X1", "sh"], ["y"])
        # Must not raise AssertionError.
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("y"), TFLOAT)
        # Shape inference falls back when ranks are incompatible.
        self.assertFalse(b.has_shape("y"))


class TestMakeNodeSetTypeShapeConstantConstantOfShape(ExtTestCase):
    def test_constant_of_shape_static(self):
        b = BasicShapeBuilder()
        cst = onh.from_array(np.array([2, 3], dtype=np.int64), name="sh")
        b.set_constant("sh", cst)
        node = oh.make_node("ConstantOfShape", ["sh"], ["y"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("y"), TFLOAT)
        self.assertEqual(b.get_shape("y"), (2, 3))

    def test_constant_of_shape_explicit_type(self):
        b = BasicShapeBuilder()
        cst = onh.from_array(np.array([2, 3], dtype=np.int64), name="sh")
        b.set_constant("sh", cst)
        value_tensor = oh.make_tensor("value", TINT64, [1], [0])
        node = oh.make_node("ConstantOfShape", ["sh"], ["y"], value=value_tensor)
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("y"), TINT64)


class TestMakeNodeSetTypeShapeConstantGatherElements(ExtTestCase):
    def test_gather_elements_shape_from_indices(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3, 4))
        b.set_type("idx", TINT64)
        b.set_shape("idx", (2, 4))
        node = oh.make_node("GatherElements", ["x", "idx"], ["y"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_type("y"), TFLOAT)
        self.assertEqual(b.get_shape("y"), (2, 4))

    def test_gather_elements_rank_from_indices(self):
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_rank("x", 2)
        b.set_type("idx", TINT64)
        b.set_rank("idx", 2)
        node = oh.make_node("GatherElements", ["x", "idx"], ["y"])
        b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(b.get_rank("y"), 2)


class TestComputeConstant(ExtTestCase):
    def test_compute_constant_from_tensor_proto(self):
        b = BasicShapeBuilder()
        cst = onh.from_array(np.array([1, 2, 3], dtype=np.int64), name="c")
        b.constants_["c"] = cst
        result, _ = b.compute_constant("c")
        self.assertEqualArray(result, np.array([1, 2, 3], dtype=np.int64))

    def test_compute_constant_identity(self):
        b = BasicShapeBuilder()
        # register input constant
        x_val = onh.from_array(np.array([1.0, 2.0], dtype=np.float32), name="x")
        b.constants_["x"] = x_val
        # identity node
        node = oh.make_node("Identity", ["x"], ["y"])
        b.constants_["y"] = node
        b.set_type("y", TFLOAT)
        result, _ = b.compute_constant("y")
        self.assertEqualArray(result, np.array([1.0, 2.0], dtype=np.float32))

    def test_compute_constant_reshape(self):
        b = BasicShapeBuilder()
        x_val = onh.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), name="x")
        sh_val = onh.from_array(np.array([3, 2], dtype=np.int64), name="sh")
        b.constants_["x"] = x_val
        b.constants_["sh"] = sh_val
        node = oh.make_node("Reshape", ["x", "sh"], ["y"])
        b.constants_["y"] = node
        b.set_type("y", TFLOAT)
        result, _ = b.compute_constant("y")
        self.assertEqual(result.shape, (3, 2))

    def test_compute_constant_add(self):
        b = BasicShapeBuilder()
        a_val = onh.from_array(np.array([1, 2], dtype=np.int64), name="a")
        c_val = onh.from_array(np.array([3, 4], dtype=np.int64), name="c")
        b.constants_["a"] = a_val
        b.constants_["c"] = c_val
        node = oh.make_node("Add", ["a", "c"], ["y"])
        b.constants_["y"] = node
        b.set_type("y", TINT64)
        b.set_shape("a", (2,))
        b.set_type("a", TINT64)
        b.set_shape("c", (2,))
        b.set_type("c", TINT64)
        result, _ = b.compute_constant("y")
        self.assertEqualArray(result, np.array([4, 6], dtype=np.int64))

    def test_compute_constant_shape_node_no_shape(self):
        # When input has no shape, compute_constant for a Shape node returns (None, None)
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        node = oh.make_node("Shape", ["x"], ["s"])
        b.constants_["s"] = node
        result, feeds = b.compute_constant("s", exc=False)
        self.assertIsNone(result)
        self.assertIsNone(feeds)

    def test_compute_constant_shape_node_static_shape(self):
        # When input has a fully static shape, Shape node returns it as int64 array
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (3, 4))
        node = oh.make_node("Shape", ["x"], ["s"])
        b.constants_["s"] = node
        result, feeds = b.compute_constant("s")
        self.assertEqualArray(result, np.array([3, 4], dtype=np.int64))
        self.assertIn("x", feeds)

    def test_compute_constant_shape_node_static_shape_with_start_end(self):
        # Shape node with start/end attributes on a static shape
        b = BasicShapeBuilder()
        b.set_type("x", TFLOAT)
        b.set_shape("x", (2, 3, 4, 5))
        node = oh.make_node("Shape", ["x"], ["s"], start=1, end=3)
        b.constants_["s"] = node
        result, feeds = b.compute_constant("s")
        self.assertEqualArray(result, np.array([3, 4], dtype=np.int64))
        self.assertIn("x", feeds)

    def test_compute_constant_not_a_constant_raises(self):
        b = BasicShapeBuilder()
        self.assertRaises(AssertionError, b.compute_constant, "nonexistent")

    def test_compute_constant_via_run_model(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Shape", ["X"], ["s"])],
                "g",
                [_mkv_("X", TFLOAT, [3, 4])],
                [_mkv_("s", TINT64, [2])],
            )
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        # Shape node output is marked as a constant and has correct shape/type
        self.assertTrue(b.is_constant("s"))
        self.assertEqual(b.get_type("s"), TINT64)
        self.assertEqual(b.get_shape("s"), (2,))


class TestMakeNodeSetTypeShapeConstantConstant(ExtTestCase):
    def test_constant_sets_shape_and_type(self):
        b = BasicShapeBuilder()
        value = oh.make_tensor("value", TFLOAT, [2, 3], [1.0] * 6)
        node = oh.make_node("Constant", [], ["c"], value=value)
        result = b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(result, (2, 3))
        self.assertEqual(b.get_shape("c"), (2, 3))
        self.assertEqual(b.get_type("c"), TFLOAT)

    def test_constant_marks_output_as_constant(self):
        b = BasicShapeBuilder()
        value = oh.make_tensor("value", TFLOAT, [2, 3], [1.0] * 6)
        node = oh.make_node("Constant", [], ["c"], value=value)
        b._make_node_set_type_shape_constant(node, {})
        self.assertTrue(b.is_constant("c"))

    def test_constant_adds_doc_string_tag(self):
        b = BasicShapeBuilder()
        value = oh.make_tensor("value", TFLOAT, [2, 3], [1.0] * 6)
        node = oh.make_node("Constant", [], ["c"], value=value)
        b._make_node_set_type_shape_constant(node, {})
        self.assertIn(":constant-3:", node.doc_string)

    def test_constant_int64_tensor(self):
        b = BasicShapeBuilder()
        value = oh.make_tensor("value", TINT64, [4], [1, 2, 3, 4])
        node = oh.make_node("Constant", [], ["c"], value=value)
        result = b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(result, (4,))
        self.assertEqual(b.get_shape("c"), (4,))
        self.assertEqual(b.get_type("c"), TINT64)

    def test_constant_scalar_tensor(self):
        b = BasicShapeBuilder()
        value = oh.make_tensor("value", TFLOAT, [], [3.14])
        node = oh.make_node("Constant", [], ["c"], value=value)
        result = b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(result, ())
        self.assertEqual(b.get_shape("c"), ())
        self.assertEqual(b.get_type("c"), TFLOAT)

    def test_constant_value_float_attribute(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Constant", [], ["c"], value_float=1.5)
        result = b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(result, ())
        self.assertEqual(b.get_shape("c"), ())
        self.assertEqual(b.get_type("c"), TFLOAT)

    def test_constant_value_int_attribute(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Constant", [], ["c"], value_int=42)
        result = b._make_node_set_type_shape_constant(node, {})
        self.assertEqual(result, ())
        self.assertEqual(b.get_shape("c"), ())
        self.assertEqual(b.get_type("c"), TINT64)


class TestMakeNodeSetTypeShapeConstantCustomDomain(ExtTestCase):
    def test_custom_domain_returns_none(self):
        b = BasicShapeBuilder()
        node = oh.make_node("MyOp", ["x"], ["y"], domain="custom.domain")
        result = b._make_node_set_type_shape_constant(node, {})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
