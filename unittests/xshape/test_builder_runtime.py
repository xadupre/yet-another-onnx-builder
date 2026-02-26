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
            node,
            {
                "data": data,
                "starts": starts,
                "ends": ends,
                "axes": axes,
                "steps": steps,
            },
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
