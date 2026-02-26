import unittest
import numpy as np
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape.shape_builder_impl import BasicShapeBuilder


class _TorchShapeBuilder(BasicShapeBuilder):
    """BasicShapeBuilder extended with a ``torch`` property for runtime tests."""

    @property
    def torch(self):
        import torch

        return torch

    def make_torch_tensor_from_np_array(self, x):
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
            RuntimeError,
            self.b._apply_slice_to_shape,
            (10, 4, 5),
            [0, slice(1, 3)],
            [0],
            [],
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

    def test_squeeze_torch_no_axis(self):
        import torch

        node = oh.make_node("Squeeze", ["x"], ["y"])
        x = torch.ones(1, 3, 1, 5, dtype=torch.float32)
        result = self.b._apply_squeeze(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 5))

    def test_squeeze_torch_with_axis_scalar(self):
        import torch

        node = oh.make_node("Squeeze", ["x", "axes"], ["y"])
        x = torch.ones(3, 1, 5, dtype=torch.float32)
        axes = np.array(1, dtype=np.int64)
        result = self.b._apply_squeeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (3, 5))

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

    def test_unsqueeze_torch_scalar_axis(self):
        import torch

        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        axes = np.array(0, dtype=np.int64)
        result = self.b._apply_unsqueeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (1, 2, 3))

    def test_unsqueeze_torch_single_axis(self):
        import torch

        node = oh.make_node("Unsqueeze", ["x", "axes"], ["y"])
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        axes = np.array([1], dtype=np.int64)
        result = self.b._apply_unsqueeze(node, {"x": x, "axes": axes})
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (2, 1, 3))

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

    def test_cast_torch_to_float32(self):
        import torch

        node = oh.make_node("Cast", ["x"], ["y"], to=1)  # 1 = FLOAT
        x = torch.arange(6, dtype=torch.int64).reshape(2, 3)
        result = self.b._apply_cast(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].dtype, torch.float32)
        self.assertEqual(tuple(result[0].shape), (2, 3))

    def test_cast_torch_to_int64(self):
        import torch

        node = oh.make_node("Cast", ["x"], ["y"], to=7)  # 7 = INT64
        x = torch.tensor([1.5, 2.7, 3.1], dtype=torch.float32)
        result = self.b._apply_cast(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].dtype, torch.int64)


class TestApplyUnaryFunction(ExtTestCase):
    def setUp(self):
        self.b = _TorchShapeBuilder()

    def test_sqrt_numpy(self):
        node = oh.make_node("Sqrt", ["x"], ["y"])
        x = np.array([4.0, 9.0, 16.0], dtype=np.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(result[0], np.array([2.0, 3.0, 4.0], dtype=np.float32))

    def test_sqrt_torch(self):
        import torch

        node = oh.make_node("Sqrt", ["x"], ["y"])
        x = torch.tensor([4.0, 9.0, 16.0], dtype=torch.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(
            result[0].numpy(), np.array([2.0, 3.0, 4.0], dtype=np.float32)
        )

    def test_exp_torch(self):
        import torch

        node = oh.make_node("Exp", ["x"], ["y"])
        x = torch.zeros(3, dtype=torch.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(result[0].numpy(), np.ones(3, dtype=np.float32))


class TestApplyUnaryFunction(ExtTestCase):
    def setUp(self):
        self.b = _TorchShapeBuilder()

    # --- numpy path ---

    def test_sqrt_numpy(self):
        node = oh.make_node("Sqrt", ["x"], ["y"])
        x = np.array([4.0, 9.0, 16.0], dtype=np.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(result[0], np.array([2.0, 3.0, 4.0], dtype=np.float32))

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
        node = oh.make_node("Log", ["x"], ["y"])
        x = np.array([1.0, 2.0], dtype=np.float32)
        self.assertRaises(AssertionError, self.b._apply_unary_function, node, {"x": x})

    # --- torch path ---

    def test_sqrt_torch(self):
        import torch

        node = oh.make_node("Sqrt", ["x"], ["y"])
        x = torch.tensor([4.0, 9.0, 16.0], dtype=torch.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(
            result[0].numpy(), np.array([2.0, 3.0, 4.0], dtype=np.float32)
        )

    def test_exp_torch(self):
        import torch

        node = oh.make_node("Exp", ["x"], ["y"])
        x = torch.tensor([0.0, 1.0], dtype=torch.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(
            result[0].numpy(), np.exp(np.array([0.0, 1.0], dtype=np.float32)), atol=1e-6
        )

    def test_reciprocal_torch(self):
        import torch

        node = oh.make_node("Reciprocal", ["x"], ["y"])
        x = torch.tensor([2.0, 4.0], dtype=torch.float32)
        result = self.b._apply_unary_function(node, {"x": x})
        self.assertEqual(len(result), 1)
        self.assertEqualArray(
            result[0].numpy(), np.array([0.5, 0.25], dtype=np.float32)
        )

    def test_unknown_op_torch_raises(self):
        import torch

        node = oh.make_node("Log", ["x"], ["y"])
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        self.assertRaises(AssertionError, self.b._apply_unary_function, node, {"x": x})


if __name__ == "__main__":
    unittest.main(verbosity=2)
