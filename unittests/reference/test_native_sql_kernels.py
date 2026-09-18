"""Exercises native runtime boundaries used by SQL conversion."""

import unittest
import numpy
from onnx_light.onnx import TensorProto, helper, numpy_helper
from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator


class TestNativeSqlKernels(ExtTestCase):
    def test_string_initializers_repeat_and_override(self):
        """Preserves string payloads across runs without changing caller data."""
        values = numpy.array(["\u00e9t\u00e9", "", "X"], dtype=object)
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Equal", ["X", "labels"], ["Y"])],
                "string_initializers",
                [helper.make_tensor_value_info("X", TensorProto.STRING, [3])],
                [helper.make_tensor_value_info("Y", TensorProto.BOOL, [3])],
                [numpy_helper.from_array(values, "labels")],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )
        serialized = model.SerializeToString()
        evaluator = ExtendedReferenceEvaluator(model)
        for inputs in (values, values[::-1].copy(), values):
            feeds = {"X": inputs}
            self.assertEqualArray(numpy.equal(inputs, values), evaluator.run(None, feeds)[0])
            self.assertEqual(set(feeds), {"X"})
        self.assertEqualArray(
            numpy.ones(3, dtype=numpy.bool_),
            evaluator.run(None, {"X": values[::-1], "labels": values[::-1]})[0],
        )
        self.assertEqual(serialized, model.SerializeToString())

    def test_less_float64(self):
        """Retains differences that disappear when narrowed to float32."""
        node = helper.make_node("Less", ["X", "Y"], ["Z"])
        x = numpy.array([1.0, 1.0 + 1e-12], dtype=numpy.float64)
        y = numpy.array([1.0 + 1e-12, 1.0], dtype=numpy.float64)
        result = ExtendedReferenceEvaluator(node).run(None, {"X": x, "Y": y})[0]
        self.assertEqualArray(numpy.less(x, y), result)

    def test_compress_strings(self):
        """Handles string selection, empty outputs and the flattening default."""
        data = numpy.array([["A", "B"], ["C", "D"]], dtype=object)
        for axis, condition in (
            (None, [True, False, True, False]),
            (0, [False, False]),
            (1, [False, True]),
        ):
            with self.subTest(axis=axis, condition=condition):
                attributes = {} if axis is None else {"axis": axis}
                node = helper.make_node("Compress", ["X", "C"], ["Y"], **attributes)
                condition = numpy.array(condition, dtype=numpy.bool_)
                result = ExtendedReferenceEvaluator(node).run(None, {"X": data, "C": condition})[
                    0
                ]
                numpy.testing.assert_array_equal(
                    numpy.compress(condition, data, axis=axis), result
                )

    def test_reduce_float64_axes(self):
        """Preserves dtype, precision, axes and keepdims for each reduction."""
        data = numpy.array([[1.0, 1.0 + 1e-10], [2.0, 2.0 + 1e-10]])
        for op_type, operation in (
            ("ReduceMin", numpy.min),
            ("ReduceMax", numpy.max),
            ("ReduceMean", numpy.mean),
        ):
            for keepdims in (0, 1):
                for opset in (13, 18):
                    with self.subTest(op_type=op_type, keepdims=keepdims, opset=opset):
                        inputs = ["X", "axes"] if opset >= 18 else ["X"]
                        attributes = {} if opset >= 18 else {"axes": [-1]}
                        node = helper.make_node(
                            op_type, inputs, ["Y"], keepdims=keepdims, **attributes
                        )
                        feeds = {"X": data}
                        if opset >= 18:
                            feeds["axes"] = numpy.array([-1], dtype=numpy.int64)
                        result = ExtendedReferenceEvaluator(node, opsets={"": opset}).run(
                            None, feeds
                        )[0]
                        self.assertEqual(result.dtype, data.dtype)
                        self.assertEqualArray(
                            operation(data, axis=-1, keepdims=bool(keepdims)),
                            result,
                            atol=0,
                            rtol=0,
                        )

    def test_reduce_empty_axes(self):
        """Distinguishes reduce-all from no-op for explicitly empty axes."""
        data = numpy.arange(6, dtype=numpy.float64).reshape(2, 3)
        for op_type, operation in (
            ("ReduceMin", numpy.min),
            ("ReduceMax", numpy.max),
            ("ReduceMean", numpy.mean),
        ):
            for noop in (0, 1):
                with self.subTest(op_type=op_type, noop=noop):
                    node = helper.make_node(
                        op_type, ["X", "axes"], ["Y"], keepdims=0, noop_with_empty_axes=noop
                    )
                    result = ExtendedReferenceEvaluator(node).run(
                        None, {"X": data, "axes": numpy.array([], dtype=numpy.int64)}
                    )[0]
                    self.assertEqualArray(
                        data if noop else numpy.asarray(operation(data)), result
                    )

    def test_reduce_empty_data(self):
        """Returns the proper neutral values for empty extremum reductions."""
        for dtype in (numpy.float64, numpy.int64, numpy.bool_):
            data = numpy.empty((0, 3), dtype=dtype)
            for op_type in ("ReduceMin", "ReduceMax"):
                minimum = op_type == "ReduceMin"
                if dtype == numpy.float64:
                    neutral = numpy.inf if minimum else -numpy.inf
                elif dtype == numpy.int64:
                    neutral = numpy.iinfo(dtype).max if minimum else numpy.iinfo(dtype).min
                else:
                    neutral = minimum
                with self.subTest(dtype=dtype, op_type=op_type):
                    node = helper.make_node(op_type, ["X", "axes"], ["Y"], keepdims=0)
                    result = ExtendedReferenceEvaluator(node).run(
                        None, {"X": data, "axes": numpy.array([0], dtype=numpy.int64)}
                    )[0]
                    self.assertEqualArray(numpy.full((3,), neutral, dtype=dtype), result)

    def test_reduce_nan_and_invalid_axes(self):
        """Propagates NaNs and rejects out-of-range reduction axes."""
        data = numpy.array([[1.0, numpy.nan], [2.0, 3.0]], dtype=numpy.float64)
        for op_type in ("ReduceMin", "ReduceMax", "ReduceMean"):
            with self.subTest(op_type=op_type):
                node = helper.make_node(op_type, ["X", "axes"], ["Y"], keepdims=0)
                evaluator = ExtendedReferenceEvaluator(node)
                result = evaluator.run(
                    None, {"X": data, "axes": numpy.array([1], dtype=numpy.int64)}
                )[0]
                self.assertTrue(numpy.isnan(result[0]))
                with self.assertRaises(numpy.exceptions.AxisError):
                    evaluator.run(None, {"X": data, "axes": numpy.array([2], dtype=numpy.int64)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
