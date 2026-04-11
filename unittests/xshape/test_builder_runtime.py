import unittest
import numpy as np
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase, requires_torch
from yobx.xshape import BasicShapeBuilder


class _TorchShapeBuilder(BasicShapeBuilder):
    """BasicShapeBuilder extended with a ``torch`` property for runtime tests."""

    def make_torch_tensor_from_np_array(self, x):
        assert self._has_torch, "torch is not available"
        import torch

        return torch.from_numpy(x)


class TestApplySliceToShape(ExtTestCase):
    def setUp(self):
        self.b = BasicShapeBuilder()

    def test_slice_first_axis(self):
        result = self.b._apply_slice_to_shape((10, 4, 5), [slice(0, 3)], [0], [])
        self.assertEqual(result, (3, 4, 5))

    def test_slice_negative_axis(self):
        # axis=-1 refers to the last dimension (size 5)
        result = self.b._apply_slice_to_shape((3, 4, 5), [slice(0, 2)], [-1], [])
        self.assertEqual(result, (3, 4, 2))

    def test_slice_middle_axis(self):
        # slicing axis=1 fills axis=0 from the original shape first
        result = self.b._apply_slice_to_shape((3, 10, 5), [slice(2, 8)], [1], [])
        self.assertEqual(result, (3, 6, 5))

    def test_slice_with_step(self):
        result = self.b._apply_slice_to_shape((10, 4, 5), [slice(0, 10, 2)], [0], [])
        self.assertEqual(result, (5, 4, 5))

    def test_slice_start_none(self):
        # start=None defaults to 0
        result = self.b._apply_slice_to_shape((10, 4, 5), [slice(None, 5)], [0], [])
        self.assertEqual(result, (5, 4, 5))

    def test_slice_stop_none(self):
        # stop=None defaults to the dimension size
        result = self.b._apply_slice_to_shape((10, 4, 5), [slice(3, None)], [0], [])
        self.assertEqual(result, (7, 4, 5))

    def test_slice_dim_clamped_to_zero(self):
        # start > stop yields a negative diff, clamped to 0
        result = self.b._apply_slice_to_shape((3, 4, 5), [slice(5, 2)], [0], [])
        self.assertEqual(result, (0, 4, 5))

    def test_multiple_slices(self):
        result = self.b._apply_slice_to_shape((10, 8, 5), [slice(0, 4), slice(2, 6)], [0, 1], [])
        self.assertEqual(result, (4, 4, 5))

    def test_integer_indices(self):
        # all_int(indices) path: len(indices) becomes the new first dimension
        result = self.b._apply_slice_to_shape((10, 4, 5), [0, 2, 5], [0], [])
        self.assertEqual(result, (3, 4, 5))

    def test_expand_axes(self):
        # expand_axes inserts a size-1 dimension at the given position
        result = self.b._apply_slice_to_shape((3, 4, 5), [slice(0, 2)], [0], [1])
        self.assertEqual(result, (2, 1, 4, 5))

    def test_mixed_raises_runtime_error(self):
        # mixing int and slice indices is unsupported
        self.assertRaises(
            RuntimeError, self.b._apply_slice_to_shape, (10, 4, 5), [0, slice(1, 3)], [0], []
        )

    def test_remaining_dims_appended(self):
        # only axis 0 is sliced; axes 1 and 2 are copied from the original shape
        result = self.b._apply_slice_to_shape((6, 4, 5), [slice(1, 4)], [0], [])
        self.assertEqual(result, (3, 4, 5))


class TestApplyTranspose(ExtTestCase):
    def setUp(self):
        self.b = _TorchShapeBuilder()

    def test_transpose_2d(self):
        node = oh.make_node("Transpose", ["x"], ["y"], perm=[1, 0])
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        result = self.b._apply_transpose(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 2))

    def test_transpose_3d(self):
        node = oh.make_node("Transpose", ["x"], ["y"], perm=[2, 0, 1])
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        result = self.b._apply_transpose(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (4, 2, 3))

    @requires_torch()
    def test_transpose_torch(self):
        import torch

        node = oh.make_node("Transpose", ["x"], ["y"], perm=[1, 0])
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        result = self.b._apply_transpose(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 2))


class TestApplyExpand(ExtTestCase):
    def setUp(self):
        self.b = _TorchShapeBuilder()

    def test_expand_numpy(self):
        node = oh.make_node("Expand", ["x", "shape"], ["y"])
        x = np.ones((1, 3), dtype=np.float32)
        new_shape = np.array([2, 3], dtype=np.int64)
        result = self.b._apply_expand(node, {"x": x, "shape": new_shape})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (2, 3))

    @requires_torch()
    def test_expand_torch(self):
        import torch

        node = oh.make_node("Expand", ["x", "shape"], ["y"])
        x = torch.ones(1, 3)
        new_shape = np.array([2, 3], dtype=np.int64)
        result = self.b._apply_expand(node, {"x": x, "shape": new_shape})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (2, 3))

    def test_expand_broadcast(self):
        node = oh.make_node("Expand", ["x", "shape"], ["y"])
        x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        new_shape = np.array([3, 4], dtype=np.int64)
        result = self.b._apply_expand(node, {"x": x, "shape": new_shape})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (3, 4))


class TestApplySqueeze(ExtTestCase):
    def setUp(self):
        self.b = BasicShapeBuilder()

    def test_squeeze_with_axis_scalar(self):
        node = oh.make_node("Squeeze", ["x", "axes"], ["y"])
        x = np.ones((3, 1, 5), dtype=np.float32)
        axes = np.array(1, dtype=np.int64)
        result = self.b._apply_squeeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (3, 5))

    def test_squeeze_with_axis_array(self):
        node = oh.make_node("Squeeze", ["x", "axes"], ["y"])
        x = np.ones((1, 3, 1, 5), dtype=np.float32)
        axes = np.array([0, 2], dtype=np.int64)
        result = self.b._apply_squeeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (3, 5))

    def test_squeeze_no_axis(self):
        node = oh.make_node("Squeeze", ["x"], ["y"])
        x = np.ones((1, 3, 1, 5), dtype=np.float32)
        result = self.b._apply_squeeze(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (3, 5))

    @requires_torch()
    def test_squeeze_torch_no_axis(self):
        import torch

        node = oh.make_node("Squeeze", ["x"], ["y"])
        x = torch.ones(1, 3, 1, 5, dtype=torch.float32)
        result = self.b._apply_squeeze(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 5))

    @requires_torch()
    def test_squeeze_torch_with_axis_scalar(self):
        import torch

        node = oh.make_node("Squeeze", ["x", "axes"], ["y"])
        x = torch.ones(3, 1, 5, dtype=torch.float32)
        axes = np.array(1, dtype=np.int64)
        result = self.b._apply_squeeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 5))

    @requires_torch()
    def test_squeeze_torch_with_axis_array(self):
        import torch

        node = oh.make_node("Squeeze", ["x", "axes"], ["y"])
        x = torch.ones(1, 3, 1, 5, dtype=torch.float32)
        axes = np.array([0, 2], dtype=np.int64)
        result = self.b._apply_squeeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 5))


class TestApplyUnsqueeze(ExtTestCase):
    def setUp(self):
        self.b = BasicShapeBuilder()

    def test_unsqueeze_scalar_axis(self):
        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        axes = np.array(0, dtype=np.int64)
        result = self.b._apply_unsqueeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (1, 2, 3))

    def test_unsqueeze_array_axis_single(self):
        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        axes = np.array([1], dtype=np.int64)
        result = self.b._apply_unsqueeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (2, 1, 3))

    def test_unsqueeze_multiple_axes(self):
        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        axes = np.array([0, 3], dtype=np.int64)
        result = self.b._apply_unsqueeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (1, 2, 3, 1))

    @requires_torch()
    def test_unsqueeze_torch_scalar_axis(self):
        import torch

        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        axes = np.array(0, dtype=np.int64)
        result = self.b._apply_unsqueeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (1, 2, 3))

    @requires_torch()
    def test_unsqueeze_torch_single_axis(self):
        import torch

        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        axes = np.array([1], dtype=np.int64)
        result = self.b._apply_unsqueeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (2, 1, 3))

    @requires_torch()
    def test_unsqueeze_torch_multiple_axes(self):
        import torch

        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        axes = np.array([0, 3], dtype=np.int64)
        result = self.b._apply_unsqueeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (1, 2, 3, 1))


class TestApplyCast(ExtTestCase):
    def setUp(self):
        self.b = _TorchShapeBuilder()

    def test_cast_numpy_to_float32(self):
        node = oh.make_node("Cast", ["x"], ["y"], to=1)  # 1 = FLOAT
        x = np.arange(6, dtype=np.int64).reshape(2, 3)
        result = self.b._apply_cast(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (2, 3))

    @requires_torch()
    def test_cast_torch_to_float32(self):
        import torch

        node = oh.make_node("Cast", ["x"], ["y"], to=1)  # 1 = FLOAT
        x = torch.arange(6, dtype=torch.int64).reshape(2, 3)
        result = self.b._apply_cast(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].dtype, torch.float32)
        self.assertEqual(tuple(result[0].shape), (2, 3))

    @requires_torch()
    def test_cast_torch_to_int64(self):
        import torch

        node = oh.make_node("Cast", ["x"], ["y"], to=7)  # 7 = INT64
        x = torch.tensor([1.5, 2.7, 3.1], dtype=torch.float32)
        result = self.b._apply_cast(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].dtype, torch.int64)


class TestApplyShapeOnShape(ExtTestCase):
    def setUp(self):
        self.b = _TorchShapeBuilder()

    def test_no_attributes_returns_full_shape(self):
        node = oh.make_node("Shape", ["x"], ["s"])
        result = self.b._apply_shape_on_shape(node, (2, 3, 4))
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].tolist()), (2, 3, 4))

    def test_start_attribute_slices_from_start(self):
        node = oh.make_node("Shape", ["x"], ["s"], start=1)
        result = self.b._apply_shape_on_shape(node, (2, 3, 4))
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].tolist()), (3, 4))

    def test_end_attribute_slices_to_end(self):
        node = oh.make_node("Shape", ["x"], ["s"], end=2)
        result = self.b._apply_shape_on_shape(node, (2, 3, 4))
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].tolist()), (2, 3))

    def test_start_and_end_attributes(self):
        node = oh.make_node("Shape", ["x"], ["s"], start=1, end=3)
        result = self.b._apply_shape_on_shape(node, (2, 3, 4, 5))
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].tolist()), (3, 4))

    @requires_torch()
    def test_result_dtype_is_int64(self):
        import torch

        node = oh.make_node("Shape", ["x"], ["s"])
        result = self.b._apply_shape_on_shape(node, (2, 3))
        self.assertEqual(result[0].dtype, torch.int64)


class TestApplyShape(ExtTestCase):
    def setUp(self):
        self.b = _TorchShapeBuilder()

    def test_shape_of_numpy_array(self):
        node = oh.make_node("Shape", ["x"], ["s"])
        x = np.zeros((2, 3, 4), dtype=np.float32)
        result = self.b._apply_shape(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].tolist()), (2, 3, 4))

    @requires_torch()
    def test_shape_of_torch_tensor(self):
        import torch

        node = oh.make_node("Shape", ["x"], ["s"])
        x = torch.zeros(5, 6)
        result = self.b._apply_shape(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].tolist()), (5, 6))

    def test_shape_with_start_attribute(self):
        node = oh.make_node("Shape", ["x"], ["s"], start=1)
        x = np.zeros((2, 3, 4), dtype=np.float32)
        result = self.b._apply_shape(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].tolist()), (3, 4))

    def test_shape_with_start_end_attributes(self):
        node = oh.make_node("Shape", ["x"], ["s"], start=1, end=3)
        x = np.zeros((2, 3, 4, 5), dtype=np.float32)
        result = self.b._apply_shape(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].tolist()), (3, 4))


class TestApplyUnaryFunction(ExtTestCase):
    def setUp(self):
        self.b = _TorchShapeBuilder()

    # --- numpy path ---

    def test_exp_numpy(self):
        node = oh.make_node("Exp", ["x"], ["y"])
        x = np.array([0.0, 1.0], dtype=np.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(result[0], np.exp(x).astype(np.float32))

    def test_reciprocal_numpy(self):
        node = oh.make_node("Reciprocal", ["x"], ["y"])
        x = np.array([2.0, 4.0], dtype=np.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(result[0], np.array([0.5, 0.25], dtype=np.float32))

    def test_unknown_op_numpy_raises(self):
        node = oh.make_node("LogRaise", ["x"], ["y"])
        x = np.array([1.0, 2.0], dtype=np.float32)
        self.assertRaises(AssertionError, self.b._apply_unary_function, node, {"x": x})

    # --- torch path ---

    @requires_torch()
    def test_reciprocal_torch(self):
        import torch

        node = oh.make_node("Reciprocal", ["x"], ["y"])
        x = torch.tensor([2.0, 4.0], dtype=torch.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(result[0].numpy(), np.array([0.5, 0.25], dtype=np.float32))

    @requires_torch()
    def test_unknown_op_torch_log(self):
        import torch

        node = oh.make_node("Log", ["x"], ["y"])
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(result[0], np.array([0.0, 0.693147], dtype=np.float32), atol=1e-5)

    def test_sqrt_numpy(self):
        node = oh.make_node("Sqrt", ["x"], ["y"])
        x = np.array([4.0, 9.0, 16.0], dtype=np.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(result[0], np.array([2.0, 3.0, 4.0], dtype=np.float32))

    @requires_torch()
    def test_sqrt_torch(self):
        import torch

        node = oh.make_node("Sqrt", ["x"], ["y"])
        x = torch.tensor([4.0, 9.0, 16.0], dtype=torch.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(result[0].numpy(), np.array([2.0, 3.0, 4.0], dtype=np.float32))

    @requires_torch()
    def test_exp_torch(self):
        import torch

        node = oh.make_node("Exp", ["x"], ["y"])
        x = torch.zeros(3, dtype=torch.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(result[0].numpy(), np.ones(3, dtype=np.float32))


class TestApplyBinaryOp(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = _TorchShapeBuilder()

    def _make_node(self, op_type, input_names=("a", "b"), output_names=("c",)):
        return oh.make_node(op_type, list(input_names), list(output_names))

    # --- numpy tests ---

    def test_add_numpy(self):
        node = self._make_node("Add")
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        result = self.b._apply_binary_op(node, {"a": a, "b": b})
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.array([4.0, 6.0], dtype=np.float32))

    def test_mul_numpy(self):
        node = self._make_node("Mul")
        a = np.array([2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0], dtype=np.float32)
        result = self.b._apply_binary_op(node, {"a": a, "b": b})
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.array([8.0, 15.0], dtype=np.float32))

    def test_sub_numpy(self):
        node = self._make_node("Sub")
        a = np.array([5.0, 7.0], dtype=np.float32)
        b = np.array([3.0, 2.0], dtype=np.float32)
        result = self.b._apply_binary_op(node, {"a": a, "b": b})
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.array([2.0, 5.0], dtype=np.float32))

    def test_div_numpy(self):
        node = self._make_node("Div")
        a = np.array([6.0, 9.0], dtype=np.float32)
        b = np.array([2.0, 3.0], dtype=np.float32)
        result = self.b._apply_binary_op(node, {"a": a, "b": b})
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.array([3.0, 3.0], dtype=np.float32))

    def test_pow_numpy(self):
        node = self._make_node("Pow")
        a = np.array([2.0, 3.0], dtype=np.float32)
        b = np.array([3.0, 2.0], dtype=np.float32)
        result = self.b._apply_binary_op(node, {"a": a, "b": b})
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.array([8.0, 9.0], dtype=np.float32))

    # --- torch tests ---

    @requires_torch()
    def test_add_torch(self):
        import torch

        node = self._make_node("Add")
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        result = self.b._apply_binary_op(node, {"a": a, "b": b})
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result[0].tolist()), [4.0, 6.0])

    @requires_torch()
    def test_mul_torch(self):
        import torch

        node = self._make_node("Mul")
        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        result = self.b._apply_binary_op(node, {"a": a, "b": b})
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result[0].tolist()), [8.0, 15.0])

    @requires_torch()
    def test_sub_torch(self):
        import torch

        node = self._make_node("Sub")
        a = torch.tensor([5.0, 7.0])
        b = torch.tensor([3.0, 2.0])
        result = self.b._apply_binary_op(node, {"a": a, "b": b})
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result[0].tolist()), [2.0, 5.0])

    @requires_torch()
    def test_div_torch(self):
        import torch

        node = self._make_node("Div")
        a = torch.tensor([6.0, 9.0])
        b = torch.tensor([2.0, 3.0])
        result = self.b._apply_binary_op(node, {"a": a, "b": b})
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result[0].tolist()), [3.0, 3.0])

    @requires_torch()
    def test_pow_torch(self):
        import torch

        node = self._make_node("Pow")
        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([3.0, 2.0])
        result = self.b._apply_binary_op(node, {"a": a, "b": b})
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result[0].tolist()), [8.0, 9.0])

    # --- error case ---

    def test_unknown_op_raises(self):
        node = self._make_node("Xor")
        a = np.array([1.0], dtype=np.float32)
        b = np.array([1.0], dtype=np.float32)
        self.assertRaises(AssertionError, self.b._apply_binary_op, node, {"a": a, "b": b})


class TestApplySlice(ExtTestCase):
    def setUp(self):
        self.b = _TorchShapeBuilder()

    def test_slice_no_axes(self):
        # Without axes: starts/ends apply to each dimension in order
        node = oh.make_node("Slice", ["data", "starts", "ends"], ["y"])
        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        starts = np.array([1, 0], dtype=np.int64)
        ends = np.array([3, 4], dtype=np.int64)
        result = self.b._apply_slice(node, {"data": data, "starts": starts, "ends": ends})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (2, 4))

    def test_slice_with_axes(self):
        # axes=[0]: slice only the first dimension
        node = oh.make_node("Slice", ["data", "starts", "ends", "axes"], ["y"])
        data = np.arange(30, dtype=np.float32).reshape(6, 5)
        starts = np.array([1], dtype=np.int64)
        ends = np.array([4], dtype=np.int64)
        axes = np.array([0], dtype=np.int64)
        result = self.b._apply_slice(
            node, {"data": data, "starts": starts, "ends": ends, "axes": axes}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 5))

    def test_slice_with_axes_and_steps(self):
        # axes=[0], steps=[2]: every other row
        node = oh.make_node("Slice", ["data", "starts", "ends", "axes", "steps"], ["y"])
        data = np.arange(30, dtype=np.float32).reshape(6, 5)
        starts = np.array([0], dtype=np.int64)
        ends = np.array([6], dtype=np.int64)
        axes = np.array([0], dtype=np.int64)
        steps = np.array([2], dtype=np.int64)
        result = self.b._apply_slice(
            node, {"data": data, "starts": starts, "ends": ends, "axes": axes, "steps": steps}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 5))

    def test_slice_no_axes_with_steps(self):
        # Without axes but with steps
        node = oh.make_node("Slice", ["data", "starts", "ends", "", "steps"], ["y"])
        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        starts = np.array([0, 0], dtype=np.int64)
        ends = np.array([4, 5], dtype=np.int64)
        steps = np.array([2, 2], dtype=np.int64)
        result = self.b._apply_slice(
            node, {"data": data, "starts": starts, "ends": ends, "steps": steps}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (2, 3))

    def test_slice_multiple_axes(self):
        # Slice on two different axes simultaneously
        node = oh.make_node("Slice", ["data", "starts", "ends", "axes"], ["y"])
        data = np.arange(60, dtype=np.float32).reshape(6, 5, 2)
        starts = np.array([1, 2], dtype=np.int64)
        ends = np.array([4, 5], dtype=np.int64)
        axes = np.array([0, 1], dtype=np.int64)
        result = self.b._apply_slice(
            node, {"data": data, "starts": starts, "ends": ends, "axes": axes}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 3, 2))

    def test_slice_negative_axis(self):
        # axes=[-1]: slice last dimension
        node = oh.make_node("Slice", ["data", "starts", "ends", "axes"], ["y"])
        data = np.arange(30, dtype=np.float32).reshape(3, 10)
        starts = np.array([2], dtype=np.int64)
        ends = np.array([7], dtype=np.int64)
        axes = np.array([-1], dtype=np.int64)
        result = self.b._apply_slice(
            node, {"data": data, "starts": starts, "ends": ends, "axes": axes}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 5))

    @requires_torch()
    def test_slice_torch_tensor_input(self):
        import torch

        # Torch tensors pass through without numpy conversion
        node = oh.make_node("Slice", ["data", "starts", "ends", "axes"], ["y"])
        data = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        starts = torch.tensor([0], dtype=torch.int64)
        ends = torch.tensor([2], dtype=torch.int64)
        axes = torch.tensor([0], dtype=torch.int64)
        result = self.b._apply_slice(
            node, {"data": data, "starts": starts, "ends": ends, "axes": axes}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (2, 5))

    def test_slice_3d_axis1(self):
        # Slice the middle axis of a 3-D array
        node = oh.make_node("Slice", ["data", "starts", "ends", "axes"], ["y"])
        data = np.zeros((3, 10, 4), dtype=np.float32)
        starts = np.array([3], dtype=np.int64)
        ends = np.array([8], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)
        result = self.b._apply_slice(
            node, {"data": data, "starts": starts, "ends": ends, "axes": axes}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 5, 4))


class TestApplyWhere(ExtTestCase):
    def setUp(self):
        self.b_numpy = BasicShapeBuilder()
        self.b_torch = _TorchShapeBuilder()

    def test_where_numpy_1d(self):
        node = oh.make_node("Where", ["cond", "x", "y"], ["z"])
        cond = np.array([True, False, True])
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        result = self.b_numpy._apply_where(node, {"cond": cond, "x": x, "y": y})
        self.assertEqual(len(result), 1)
        expected = np.array([1.0, 20.0, 3.0], dtype=np.float32)
        self.assertEqualArray(expected, result[0])

    def test_where_numpy_2d(self):
        node = oh.make_node("Where", ["cond", "x", "y"], ["z"])
        cond = np.array([[True, False], [False, True]])
        x = np.ones((2, 2), dtype=np.float32)
        y = np.zeros((2, 2), dtype=np.float32)
        result = self.b_numpy._apply_where(node, {"cond": cond, "x": x, "y": y})
        self.assertEqual(len(result), 1)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.assertEqualArray(expected, result[0])

    @requires_torch()
    def test_where_torch_tensors(self):
        import torch

        node = oh.make_node("Where", ["cond", "x", "y"], ["z"])
        cond = torch.tensor([True, False, True])
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([10.0, 20.0, 30.0])
        result = self.b_torch._apply_where(node, {"cond": cond, "x": x, "y": y})
        self.assertEqual(len(result), 1)
        expected = torch.tensor([1.0, 20.0, 3.0])
        self.assertEqualArray(expected, result[0])

    def test_where_numpy_inputs(self):
        node = oh.make_node("Where", ["cond", "x", "y"], ["z"])
        cond = np.array([True, False, True])
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        result = self.b_torch._apply_where(node, {"cond": cond, "x": x, "y": y})
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)
        expected = np.array([1.0, 20.0, 3.0], dtype=np.float32)
        self.assertEqualArray(expected, result[0])


class TestApplyTrilu(ExtTestCase):
    def setUp(self):
        self.b = _TorchShapeBuilder()

    def test_numpy_upper_no_k(self):
        # Single input: default upper triangular with k=0
        node = oh.make_node("Trilu", ["x"], ["y"])
        x = np.arange(9, dtype=np.float32).reshape(3, 3)
        result = self.b._apply_trilu(node, {"x": x})
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.triu(x, 0))

    def test_numpy_upper_with_k0(self):
        # Two inputs: upper triangular with explicit k=0
        node = oh.make_node("Trilu", ["x", "k"], ["y"])
        x = np.arange(9, dtype=np.float32).reshape(3, 3)
        k = np.array(0, dtype=np.int64)
        result = self.b._apply_trilu(node, {"x": x, "k": k})
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.triu(x, 0))

    def test_numpy_upper_with_k1(self):
        # Two inputs: upper triangular with k=1 (exclude main diagonal)
        node = oh.make_node("Trilu", ["x", "k"], ["y"])
        x = np.arange(9, dtype=np.float32).reshape(3, 3)
        k = np.array(1, dtype=np.int64)
        result = self.b._apply_trilu(node, {"x": x, "k": k})
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.triu(x, 1))

    def test_numpy_lower_no_k(self):
        # Single input: lower triangular (upper=0) with default k=0
        node = oh.make_node("Trilu", ["x"], ["y"], upper=0)
        x = np.arange(9, dtype=np.float32).reshape(3, 3)
        result = self.b._apply_trilu(node, {"x": x})
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.tril(x, 0))

    def test_numpy_lower_with_k1(self):
        # Two inputs: lower triangular with k=1
        node = oh.make_node("Trilu", ["x", "k"], ["y"], upper=0)
        x = np.arange(9, dtype=np.float32).reshape(3, 3)
        k = np.array(1, dtype=np.int64)
        result = self.b._apply_trilu(node, {"x": x, "k": k})
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.tril(x, 1))

    @requires_torch()
    def test_torch_upper_with_k0(self):
        import torch

        # Torch upper triangular with k=tensor(0)
        node = oh.make_node("Trilu", ["x", "k"], ["y"])
        x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        k = torch.tensor(0, dtype=torch.int64)
        result = self.b._apply_trilu(node, {"x": x, "k": k})
        self.assertEqual(len(result), 1)
        expected = torch.triu(x, 0)
        self.assertEqual(tuple(result[0].shape), (3, 3))
        self.assertTrue(torch.equal(result[0], expected))

    @requires_torch()
    def test_torch_lower_with_k0(self):
        import torch

        # Torch lower triangular with k=tensor(0), upper=0
        node = oh.make_node("Trilu", ["x", "k"], ["y"], upper=0)
        x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        k = torch.tensor(0, dtype=torch.int64)
        result = self.b._apply_trilu(node, {"x": x, "k": k})
        self.assertEqual(len(result), 1)
        expected = torch.tril(x, 0)
        self.assertEqual(tuple(result[0].shape), (3, 3))
        self.assertTrue(torch.equal(result[0], expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
