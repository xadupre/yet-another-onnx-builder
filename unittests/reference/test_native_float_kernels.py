"""Exercises dtype-sensitive callbacks in the native graph runtime."""

import unittest
import numpy
from onnx_light.onnx import helper
from scipy.special import log_softmax
from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator


class TestNativeFloatKernels(ExtTestCase):
    def test_log_softmax_versions_and_dtypes(self):
        """Preserves dtype, stability and legacy flattening semantics."""
        for dtype in (numpy.float16, numpy.float32, numpy.float64):
            data = (numpy.arange(24).reshape(2, 3, 4) * 10 + 10000).astype(dtype)
            for opset in (11, 13, 18):
                for axis in (None, 0, 1, -1):
                    with self.subTest(dtype=dtype, opset=opset, axis=axis):
                        node = helper.make_node(
                            "LogSoftmax", ["X"], ["Y"], **({} if axis is None else {"axis": axis})
                        )
                        result = ExtendedReferenceEvaluator(node, opsets={"": opset}).run(
                            None, {"X": data}
                        )[0]
                        selected_axis = (1 if opset < 13 else -1) if axis is None else axis
                        values = data.astype(numpy.float64)
                        if opset < 13:
                            selected_axis %= data.ndim
                            values = values.reshape(
                                int(numpy.prod(data.shape[:selected_axis], dtype=numpy.int64)), -1
                            )
                            selected_axis = 1
                        expected = log_softmax(values, axis=selected_axis).reshape(data.shape)
                        self.assertEqual(result.dtype, data.dtype)
                        self.assertEqualArray(
                            expected.astype(dtype), result, atol=1e-6, rtol=1e-6
                        )

    def test_log_softmax_empty_and_invalid_axis(self):
        """Preserves empty output shapes and reports invalid axes."""
        for opset in (11, 18):
            for axis in (0, 1):
                with self.subTest(opset=opset, axis=axis):
                    node = helper.make_node("LogSoftmax", ["X"], ["Y"], axis=axis)
                    data = numpy.empty((2, 0), dtype=numpy.float16)
                    result = ExtendedReferenceEvaluator(node, opsets={"": opset}).run(
                        None, {"X": data}
                    )[0]
                    self.assertEqualArray(data, result)
            node = helper.make_node("LogSoftmax", ["X"], ["Y"], axis=2)
            with self.assertRaises(numpy.exceptions.AxisError):
                ExtendedReferenceEvaluator(node, opsets={"": opset}).run(
                    None, {"X": numpy.ones((2, 3), dtype=numpy.float32)}
                )

    def test_where_precision_broadcast_and_signed_zero(self):
        """Selects values without narrowing them or losing signed zeros."""
        condition = numpy.array([[True, False, True], [False, True, False]])
        for dtype in (numpy.float16, numpy.float32, numpy.float64):
            with self.subTest(dtype=dtype):
                x = numpy.array([-0.0, 1 + 1e-12, 10000], dtype=dtype)
                y = numpy.array([[0.0], [-2.0]], dtype=dtype)
                node = helper.make_node("Where", ["C", "X", "Y"], ["Z"])
                result = ExtendedReferenceEvaluator(node).run(
                    None, {"C": condition, "X": x, "Y": y}
                )[0]
                expected = numpy.where(condition, x, y)
                self.assertEqualArray(expected, result, atol=0, rtol=0)
                self.assertEqualArray(numpy.signbit(expected), numpy.signbit(result))

    def test_where_invalid_dtypes(self):
        """Rejects nonboolean conditions and mismatched input dtypes."""
        node = helper.make_node("Where", ["C", "X", "Y"], ["Z"])
        evaluator = ExtendedReferenceEvaluator(node)
        x = numpy.ones(2, dtype=numpy.float32)
        with self.assertRaisesRegex(TypeError, "boolean condition"):
            evaluator.run(None, {"C": x, "X": x, "Y": x})
        with self.assertRaisesRegex(TypeError, "matching data types"):
            evaluator.run(
                None, {"C": x.astype(numpy.bool_), "X": x, "Y": x.astype(numpy.float64)}
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
