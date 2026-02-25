import unittest
import onnx
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape.shape_builder import ShapeBuilder
from yobx.xshape.shape_type_compute import broadcast_shape, _set_shape_type_op_any_known

TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64


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
        return self._ranks.get(
            name, len(self._shapes[name]) if name in self._shapes else None
        )

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
