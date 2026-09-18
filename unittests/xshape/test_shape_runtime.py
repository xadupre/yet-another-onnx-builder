"""Checks native shape-value inference through public ONNX nodes."""

import unittest
import numpy as np
from onnx_light.onnx import TensorProto, helper, numpy_helper

from yobx.xshape import NativeShapeInference


class TestNativeShapeValues(unittest.TestCase):
    @staticmethod
    def context(values):
        """Seeds native shape-tensor literals."""
        inference = NativeShapeInference()
        for name, value in values.items():
            inference.set_constant(name, numpy_helper.from_array(np.array(value, dtype=np.int64)))
        return inference

    def test_binary_shape_arithmetic(self):
        cases = (
            ("Add", [2], [3], (5,)),
            ("Sub", [7], [2], (5,)),
            ("Mul", [2], [3], None),
            ("Div", [8], [2], None),
            ("Mod", [7], [3], None),
            ("Add", [2, 3, 4], [1, 2, 3], (3, 5, 7)),
            ("Add", [2, 3, 4], [1], (3, 4, 5)),
        )
        for op, x, y, expected in cases:
            with self.subTest(op=op, x=x, y=y):
                inference = self.context({"X": x, "Y": y})
                inference.run_node(helper.make_node(op, ["X", "Y"], ["Z"]), cost=False)
                self.assertEqual(inference.value_as_shape("Z"), expected)
                self.assertEqual(inference.get_shape("Z"), (max(len(x), len(y)),))

    def test_identity_and_abs(self):
        for op in ("Identity", "Abs"):
            with self.subTest(op=op):
                inference = self.context({"X": [2, 3, 4]})
                inference.run_node(helper.make_node(op, ["X"], ["Y"]), cost=False)
                # The wheel does not propagate shape values through these operators.
                self.assertIsNone(inference.value_as_shape("Y"))
                self.assertEqual(inference.get_shape("Y"), (3,))

    def test_symbolic_identity_and_arithmetic(self):
        inference = NativeShapeInference()
        inference.set_type("X", TensorProto.INT64)
        inference.set_shape("X", (2,))
        inference.set_value_shape("X", ("N", 3))
        inference.set_constant("one", numpy_helper.from_array(np.array([1], dtype=np.int64)))
        inference.run_node(helper.make_node("Identity", ["X"], ["Y"]), cost=False)
        inference.run_node(helper.make_node("Add", ["X", "one"], ["Z"]), cost=False)
        self.assertIsNone(inference.value_as_shape("Y"))
        self.assertEqual(inference.value_as_shape("Z"), ("N+1", 4))

    def test_gather_indices(self):
        for indices, expected in (
            (1, (3,)),
            ([1], (3,)),
            ([0, 2], (2, 4)),
            ([2, 0, 1], (4, 2, 3)),
        ):
            with self.subTest(indices=indices):
                inference = self.context({"X": [2, 3, 4], "indices": indices})
                inference.run_node(
                    helper.make_node("Gather", ["X", "indices"], ["Y"]), cost=False
                )
                self.assertEqual(inference.value_as_shape("Y"), expected)

    def test_concat(self):
        inference = self.context({"X": [2, 3], "Y": [4]})
        inference.run_node(helper.make_node("Concat", ["X", "Y"], ["Z"], axis=0), cost=False)
        self.assertEqual(inference.value_as_shape("Z"), (2, 3, 4))

    def test_slice_values(self):
        for start, end, step, expected in (
            (0, 4, 1, (2, 3, 4, 5)),
            (1, 3, 1, (3, 4)),
            (0, 4, 2, (2, 4)),
        ):
            with self.subTest(start=start, end=end, step=step):
                inference = self.context(
                    {
                        "X": [2, 3, 4, 5],
                        "start": [start],
                        "end": [end],
                        "axes": [0],
                        "step": [step],
                    }
                )
                inference.run_node(
                    helper.make_node("Slice", ["X", "start", "end", "axes", "step"], ["Y"]),
                    cost=False,
                )
                self.assertIsNone(inference.value_as_shape("Y"))
                self.assertEqual(inference.get_shape("Y"), (len(expected),))

    def test_squeeze_unsqueeze_scalar(self):
        inference = self.context({"X": [3], "axes": [0]})
        inference.run_node(helper.make_node("Squeeze", ["X", "axes"], ["Y"]), cost=False)
        inference.run_node(helper.make_node("Unsqueeze", ["Y", "axes"], ["Z"]), cost=False)
        self.assertEqual(inference.get_shape("Y"), ())
        self.assertEqual(inference.get_shape("Z"), (1,))
        self.assertEqual(inference.value_as_shape("Y"), (3,))
        self.assertEqual(inference.value_as_shape("Z"), (3,))

    def test_range(self):
        for step, expected in ((1, (0, 1, 2, 3)), (2, (0, 2))):
            with self.subTest(step=step):
                inference = self.context({"start": 0, "end": 4, "step": step})
                inference.run_node(
                    helper.make_node("Range", ["start", "end", "step"], ["Y"]), cost=False
                )
                self.assertIsNone(inference.value_as_shape("Y"))
                self.assertEqual(inference.get_shape("Y"), (len(expected),))

    def test_unknown_value_is_not_computed_in_python(self):
        inference = NativeShapeInference()
        inference.set_type("X", TensorProto.INT64)
        inference.set_shape("X", (3,))
        inference.run_node(helper.make_node("Identity", ["X"], ["Y"]), cost=False)
        self.assertIsNone(inference.value_as_shape("Y"))
        with self.assertRaises(ValueError):
            inference.run_node(helper.make_node("Unknown", ["X"], ["Z"]), cost=False)


if __name__ == "__main__":
    unittest.main()
