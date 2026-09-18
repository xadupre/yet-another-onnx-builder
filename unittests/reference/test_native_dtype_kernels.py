import unittest
import ml_dtypes
import numpy
import torch
from onnx_light.onnx import AttributeProto, TensorProto, helper, numpy_helper
from yobx.reference import ExtendedReferenceEvaluator


class TestNativeDtypeKernels(unittest.TestCase):
    dtypes = (numpy.float16, ml_dtypes.bfloat16, numpy.float32, numpy.float64, numpy.int64)

    def evaluate(self, op, *inputs, opset=18, attributes=None, initializers=False):
        """Executes an explicitly registered kernel through a native model."""
        names = [f"x{i}" if value is not None else "" for i, value in enumerate(inputs)]
        feeds = {name: value for name, value in zip(names, inputs) if name}
        output_type = (
            TensorProto.INT64
            if op == "NonZero"
            else (
                TensorProto.BOOL
                if op in {"LessOrEqual", "IsInf"}
                else helper.np_dtype_to_tensor_dtype(inputs[0].dtype)
            )
        )
        attributes = dict(attributes or {})
        empty_axes = attributes.get("axes") == []
        if empty_axes:
            del attributes["axes"]
        node = helper.make_node(op, names, ["y"], **attributes)
        if empty_axes:
            node.attribute.append(
                helper.make_attribute("axes", [], attr_type=AttributeProto.INTS)
            )
        model = helper.make_model(
            helper.make_graph(
                [node],
                op,
                [
                    helper.make_tensor_value_info(
                        name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
                    )
                    for name, value in feeds.items()
                    if not initializers
                ],
                [helper.make_tensor_value_info("y", output_type, None)],
                initializer=(
                    [numpy_helper.from_array(value, name) for name, value in feeds.items()]
                    if initializers
                    else []
                ),
            ),
            opset_imports=[helper.make_opsetid("", opset)],
        )
        return ExtendedReferenceEvaluator(model).run(None, {} if initializers else feeds)[0]

    def assert_tensor_equal(self, actual, expected):
        """Checks values, dtype, and shape independently."""
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        if actual.dtype.name == "bfloat16":
            actual = actual.astype(numpy.float32)
            expected = expected.astype(numpy.float32)
        numpy.testing.assert_array_equal(actual, expected)

    def test_reduction_axes_opsets_and_dtypes(self):
        for op, function, modern in (
            ("ReduceSum", numpy.sum, 13),
            ("ReduceProd", numpy.prod, 18),
        ):
            for dtype in self.dtypes:
                x = numpy.arange(1, 25).reshape(2, 3, 4).astype(dtype)
                for opset in (modern - 1, modern):
                    for axes in (None, [], [-1], [0, -1]):
                        for keepdims in (0, 1):
                            with self.subTest(
                                op=op, dtype=dtype, opset=opset, axes=axes, keepdims=keepdims
                            ):
                                attributes = {"keepdims": keepdims}
                                inputs = [x]
                                if axes is not None:
                                    if opset < modern:
                                        attributes["axes"] = axes
                                    else:
                                        inputs.append(numpy.array(axes, dtype=numpy.int64))
                                accumulator = (
                                    numpy.float32
                                    if x.dtype.name in {"float16", "bfloat16"}
                                    else dtype
                                )
                                with numpy.errstate(over="ignore"):
                                    expected = function(
                                        x,
                                        axis=tuple(axes) if axes else None,
                                        keepdims=bool(keepdims),
                                        dtype=accumulator,
                                    ).astype(dtype)
                                    got = self.evaluate(
                                        op, *inputs, opset=opset, attributes=attributes
                                    )
                                self.assert_tensor_equal(got, expected)

    def test_reduction_noop_empty_axes(self):
        for op in ("ReduceSum", "ReduceProd"):
            for dtype in self.dtypes:
                x = numpy.array([[1, 2], [3, 4]], dtype=dtype)
                for inputs in ((x,), (x, numpy.array([], dtype=numpy.int64))):
                    with self.subTest(op=op, dtype=dtype, inputs=len(inputs)):
                        got = self.evaluate(
                            op, *inputs, attributes={"noop_with_empty_axes": 1, "keepdims": 0}
                        )
                        self.assert_tensor_equal(got, x)
                got = self.evaluate(
                    op,
                    x,
                    numpy.array([-1], dtype=numpy.int64),
                    attributes={"noop_with_empty_axes": 1, "keepdims": 0},
                )
                expected = numpy.array([3, 7] if op == "ReduceSum" else [2, 12], dtype=dtype)
                self.assert_tensor_equal(got, expected)

    def test_reduction_empty_nan_scalar(self):
        for op, identity in (("ReduceSum", 0), ("ReduceProd", 1)):
            for dtype in self.dtypes:
                with self.subTest(op=op, dtype=dtype):
                    x = numpy.empty((2, 0, 3), dtype=dtype)
                    got = self.evaluate(
                        op, x, numpy.array([1], dtype=numpy.int64), attributes={"keepdims": 0}
                    )
                    self.assert_tensor_equal(got, numpy.full((2, 3), identity, dtype=dtype))
                    x = numpy.array(3, dtype=dtype)
                    self.assert_tensor_equal(self.evaluate(op, x), x)
                    if dtype != numpy.int64:
                        x = numpy.array([1, numpy.nan, 2], dtype=dtype)
                        self.assert_tensor_equal(
                            self.evaluate(op, x), numpy.array([numpy.nan], dtype=dtype)
                        )

    def test_reduction_integer_precision_and_initializers(self):
        for op, x, expected in (
            ("ReduceSum", [2**60 + 1, -(2**60), 7], 8),
            ("ReduceProd", [2**53 + 1, 3], (2**53 + 1) * 3),
        ):
            for initializers in (False, True):
                with self.subTest(op=op, initializers=initializers):
                    got = self.evaluate(
                        op, numpy.array(x, dtype=numpy.int64), initializers=initializers
                    )
                    self.assert_tensor_equal(got, numpy.array([expected], dtype=numpy.int64))

    def test_low_precision_reductions_against_torch(self):
        for dtype, torch_dtype in (
            (numpy.float16, torch.float16),
            (ml_dtypes.bfloat16, torch.bfloat16),
        ):
            for op, function in (("ReduceSum", torch.sum), ("ReduceProd", torch.prod)):
                with self.subTest(dtype=dtype, op=op):
                    values = [256, 1, -256] if op == "ReduceSum" else [1.01] * 100
                    x = numpy.array(values, dtype=dtype)
                    expected = (
                        function(
                            torch.tensor(x.astype(numpy.float32), dtype=torch_dtype),
                            dtype=torch.float32,
                        )
                        .float()
                        .numpy()
                        .astype(dtype)
                    )
                    got = self.evaluate(op, x, attributes={"keepdims": 0})
                    self.assert_tensor_equal(got, expected)

    def test_pow_double_broadcast_and_mixed_exponent(self):
        x = numpy.array([[1 + 1e-12], [1e100], [numpy.nan]], dtype=numpy.float64)
        for dtype in (numpy.float64, numpy.int64):
            with self.subTest(dtype=dtype):
                y = numpy.array([0, 1, 2], dtype=dtype)
                got = self.evaluate("Pow", x, y)
                self.assert_tensor_equal(got, numpy.power(x, y))
                self.assertGreater(got[0, 1], 1.0)
        x = numpy.array([2**53 + 1], dtype=numpy.int64)
        self.assert_tensor_equal(self.evaluate("Pow", x, numpy.array(1, dtype=x.dtype)), x)

    def test_less_or_equal_double_and_int64(self):
        for x, y in (
            (numpy.array([[1 + 1e-12], [1 - 1e-12], [numpy.nan]]), numpy.array([1.0, 1 + 2e-12])),
            (
                numpy.array([[2**60 + 1], [2**60 - 1]], dtype=numpy.int64),
                numpy.array([2**60], dtype=numpy.int64),
            ),
        ):
            got = self.evaluate("LessOrEqual", x, y)
            self.assert_tensor_equal(got, numpy.less_equal(x, y))
            self.assertFalse(got[0, 0])

    def test_min_max_bfloat16_variadic_nan_broadcast(self):
        x = numpy.array([[1], [numpy.nan], [-4]], dtype=ml_dtypes.bfloat16)
        y = numpy.array([0, 3, numpy.nan], dtype=x.dtype)
        z = numpy.array(2, dtype=x.dtype)
        for op, function in (("Min", numpy.minimum), ("Max", numpy.maximum)):
            with self.subTest(op=op):
                self.assert_tensor_equal(self.evaluate(op, x, y, z), function(function(x, y), z))
                self.assert_tensor_equal(self.evaluate(op, x), x)

    def test_isinf_bfloat16_flags(self):
        x = numpy.array([-numpy.inf, -1, 0, numpy.inf, numpy.nan], dtype=ml_dtypes.bfloat16)
        for negative in (0, 1):
            for positive in (0, 1):
                with self.subTest(negative=negative, positive=positive):
                    expected = numpy.array([bool(negative), False, False, bool(positive), False])
                    got = self.evaluate(
                        "IsInf",
                        x,
                        attributes={"detect_negative": negative, "detect_positive": positive},
                    )
                    self.assert_tensor_equal(got, expected)

    def test_nonzero_half_bfloat16_shapes(self):
        for dtype in (numpy.float16, ml_dtypes.bfloat16):
            for values in ([[0, -0.0, 1], [-2, numpy.nan, numpy.inf]], [], [[0, 0]]):
                x = numpy.array(values, dtype=dtype)
                with self.subTest(dtype=dtype, shape=x.shape):
                    self.assert_tensor_equal(
                        self.evaluate("NonZero", x), numpy.array(numpy.nonzero(x), numpy.int64)
                    )
            for value in (0, 1, numpy.nan):
                got = self.evaluate("NonZero", numpy.array(value, dtype=dtype))
                self.assert_tensor_equal(
                    got, numpy.empty((0, int(value != 0)), dtype=numpy.int64)
                )

    def test_clip_optional_bounds_and_nan(self):
        for dtype in (numpy.float16, ml_dtypes.bfloat16, numpy.float64, numpy.int64):
            x = numpy.array([-10, -1, 0, 1, 10], dtype=dtype)
            if dtype != numpy.int64:
                x = numpy.concatenate([x, numpy.array([numpy.nan], dtype=dtype)])
            for lower, upper in ((None, None), (-2, None), (None, 2), (-2, 2), (2, -2)):
                with self.subTest(dtype=dtype, lower=lower, upper=upper):
                    low = None if lower is None else numpy.array(lower, dtype=dtype)
                    high = None if upper is None else numpy.array(upper, dtype=dtype)
                    expected = x
                    if low is not None:
                        expected = numpy.maximum(expected, low)
                    if high is not None:
                        expected = numpy.minimum(expected, high)
                    self.assert_tensor_equal(self.evaluate("Clip", x, low, high), expected)

    def test_clip_default_limits_and_legacy_attributes(self):
        for dtype in (numpy.float16, ml_dtypes.bfloat16, numpy.float64):
            limits = ml_dtypes.finfo(dtype) if dtype == ml_dtypes.bfloat16 else numpy.finfo(dtype)
            x = numpy.array([-numpy.inf, 0, numpy.inf, numpy.nan], dtype=dtype)
            with self.subTest(dtype=dtype):
                self.assert_tensor_equal(
                    self.evaluate("Clip", x),
                    numpy.array([limits.min, 0, limits.max, numpy.nan], dtype=dtype),
                )
        x = numpy.array([-1e100, 0, 1e100], dtype=numpy.float64)
        self.assert_tensor_equal(
            self.evaluate("Clip", x, opset=10),
            numpy.clip(x, -numpy.finfo(numpy.float32).max, numpy.finfo(numpy.float32).max),
        )
        self.assert_tensor_equal(
            self.evaluate("Clip", x, opset=10, attributes={"min": -2.0, "max": 3.0}),
            numpy.clip(x, -2, 3),
        )
        x = numpy.array([2**60 + 1, 2**60 + 3], dtype=numpy.int64)
        maximum = numpy.array(2**60 + 2, dtype=x.dtype)
        self.assert_tensor_equal(
            self.evaluate("Clip", x, None, maximum), numpy.minimum(x, maximum)
        )

    def test_hard_sigmoid_half_bfloat16_against_torch(self):
        for dtype, torch_dtype in (
            (numpy.float16, torch.float16),
            (ml_dtypes.bfloat16, torch.bfloat16),
        ):
            x = numpy.array(
                [-numpy.inf, -4, -3, -1.25, 0, 0.3, 2.75, 3, 4, numpy.inf, numpy.nan], dtype=dtype
            )
            with self.subTest(dtype=dtype):
                expected = (
                    torch.nn.functional.hardsigmoid(
                        torch.tensor(x.astype(numpy.float32), dtype=torch_dtype)
                    )
                    .float()
                    .numpy()
                    .astype(dtype)
                )
                got = self.evaluate("HardSigmoid", x, attributes={"alpha": 1 / 6, "beta": 0.5})
                self.assert_tensor_equal(got, expected)
                expected = numpy.clip(x.astype(numpy.float32) * 0.2 + 0.5, 0, 1).astype(dtype)
                self.assert_tensor_equal(self.evaluate("HardSigmoid", x), expected)


if __name__ == "__main__":
    unittest.main()
