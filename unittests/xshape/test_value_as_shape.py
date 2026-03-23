import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape import BasicShapeBuilder
from yobx.xbuilder import GraphBuilder


class TestValueAsShape(ExtTestCase):
    def test_shape_optimization(self):
        model_proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["x"], ["n"], start=0, end=1),
                    oh.make_node("Shape", ["x"], ["b"], start=1, end=2),
                    oh.make_node("Concat", ["n", "b"], ["shape"], axis=0),
                    oh.make_node("Add", ["shape", "one"], ["shape1"]),
                    oh.make_node("Sub", ["shape1", "one"], ["shape2"]),
                    oh.make_node("Expand", ["x", "shape2"], ["expanded"]),
                    oh.make_node("Add", ["expanded", "y1"], ["z1"]),
                    oh.make_node("Add", ["expanded", "y2"], ["z2"]),
                    oh.make_node("Add", ["expanded", "y3"], ["z3"]),
                    oh.make_node("Add", ["z1", "z2"], ["z12"]),
                    oh.make_node("Add", ["z12", "z3"], ["z"]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 1]),
                    oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, [1, "B"]),
                    oh.make_tensor_value_info("y2", onnx.TensorProto.FLOAT, [1, "B"]),
                    oh.make_tensor_value_info("y3", onnx.TensorProto.FLOAT, [1, "B"]),
                ],
                [oh.make_tensor_value_info("z", onnx.TensorProto.FLOAT, ["N", "B"])],
                [onh.from_array(np.array([1], dtype=np.int64), "one")],
                # Explicit shape annotations on intermediate values (as produced by
                # shape inference or by the model creator).  These allow the rule to
                # verify that the Expand is redundant without tracing the exact
                # computation that produced the shape tensor.
                value_info=[
                    oh.make_tensor_value_info("expanded", onnx.TensorProto.FLOAT, ["N", 1]),
                    oh.make_tensor_value_info("z1", onnx.TensorProto.FLOAT, ["N", "B"]),
                    oh.make_tensor_value_info("z2", onnx.TensorProto.FLOAT, ["N", "B"]),
                    oh.make_tensor_value_info("z3", onnx.TensorProto.FLOAT, ["N", "B"]),
                ],
            ),
            ir_version=10,
            opset_imports=[oh.make_opsetid("", 20)],
        )
        onnx.checker.check_model(model_proto)
        bshape = BasicShapeBuilder()
        bshape.run_model(model_proto)
        self.assertEqual(bshape.value_as_shape("shape1"), ("N+1", 2))
        self.assertEqual(bshape.value_as_shape("shape2"), ("N", 1))
        self.assertEqual(bshape.get_shape("expanded"), ("N", 1))
        self.assertEqual(bshape.get_shape("z1"), ("N", "B"))

        builder = GraphBuilder(model_proto)
        self.assertEqual(builder.value_as_shape("shape1"), ("N+1", 2))
        self.assertEqual(builder.value_as_shape("shape2"), ("N", 1))
        self.assertEqual(builder.get_shape("expanded"), ("N", 1))
        self.assertEqual(builder.get_shape("z1"), ("N", "B"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
