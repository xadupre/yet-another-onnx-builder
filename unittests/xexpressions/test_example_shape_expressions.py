import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape import BasicShapeBuilder
from yobx.xexpressions.simplify_expressions import simplify_expression

TFLOAT = onnx.TensorProto.FLOAT


class TestExampleShapeExpressions(ExtTestCase):
    """Validates the logic shown in docs/examples/plot_shape_expressions.py."""

    def test_concat_symbolic_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Concat", ["X", "Y"], ["Z"], axis=1)],
                "concat_graph",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq1"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq2"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(builder.get_shape("X"), ("batch", "seq1"))
        self.assertEqual(builder.get_shape("Y"), ("batch", "seq2"))
        self.assertEqual(builder.get_shape("Z"), ("batch", "seq1+seq2"))

    def test_concat_evaluate_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Concat", ["X", "Y"], ["Z"], axis=1)],
                "concat_graph",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq1"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq2"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model)
        context = dict(batch=2, seq1=5, seq2=7)
        self.assertEqual(builder.evaluate_shape("Z", context), (2, 12))

    def test_reshape_floor_division_expression(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Reshape", ["X", "shape"], ["Xr"])],
                "reshape_graph",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"])],
                [oh.make_tensor_value_info("Xr", TFLOAT, [None, None, None, None])],
                [onh.from_array(np.array([0, 0, 2, -1], dtype=np.int64), name="shape")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(builder.get_shape("Xr"), ("a", "b", 2, "c//2"))

    def test_split_ceil_division_expression(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["X", "Y"], ["xy"], axis=1),
                    oh.make_node("Split", ["xy"], ["S1", "S2"], axis=1, num_outputs=2),
                ],
                "split_graph",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "c"]),
                ],
                [
                    oh.make_tensor_value_info("S1", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("S2", TFLOAT, [None, None]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(builder.get_shape("xy"), ("a", "b+c"))
        self.assertEqual(builder.get_shape("S1"), ("a", "(1+b+c)//2"))
        self.assertEqual(builder.get_shape("S2"), ("a", "b+c-(1+b+c)//2"))
        context = dict(a=3, b=4, c=6)
        self.assertEqual(builder.evaluate_shape("S1", context), (3, 5))
        self.assertEqual(builder.evaluate_shape("S2", context), (3, 5))

    def test_simplify_expressions(self):
        self.assertEqual(simplify_expression("d + f - f"), "d")
        self.assertEqual(simplify_expression("2 * seq // 2"), "seq")
        self.assertEqual(simplify_expression("1024 * a // 2"), "512*a")
        self.assertEqual(simplify_expression("b + a"), "a+b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
