import unittest
import onnx
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape._onnx_helper import (
    element_wise_binary_op_types,
    element_wise_op_cmp_types,
    unary_like_op_types,
    str_tensor_proto_type,
    enumerate_subgraphs,
    overwrite_shape_in_model_proto,
)

TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64
_mkv_ = oh.make_tensor_value_info


class TestOnnxHelper(ExtTestCase):
    def test_element_wise_binary_op_types(self):
        types = element_wise_binary_op_types()
        self.assertIsInstance(types, set)
        self.assertIn("Add", types)
        self.assertIn("Mul", types)
        self.assertIn("Sub", types)
        self.assertIn("Div", types)

    def test_element_wise_op_cmp_types(self):
        types = element_wise_op_cmp_types()
        self.assertIsInstance(types, set)
        self.assertIn("Equal", types)
        self.assertIn("Greater", types)
        self.assertIn("Less", types)

    def test_unary_like_op_types(self):
        types = unary_like_op_types()
        self.assertIsInstance(types, set)
        self.assertIn("Relu", types)
        self.assertIn("Sigmoid", types)
        self.assertIn("Cast", types)
        self.assertIn("Exp", types)
        self.assertIn("Sqrt", types)
        self.assertIn("CumSum", types)
        self.assertIn("HardSigmoid", types)
        self.assertIn("HardSwish", types)
        self.assertIn("LeakyRelu", types)
        self.assertIn("LpNormalization", types)
        self.assertIn("LRN", types)
        self.assertIn("MeanVarianceNormalization", types)
        self.assertIn("Mish", types)
        self.assertIn("Shrink", types)
        self.assertIn("Trilu", types)

    def test_str_tensor_proto_type(self):
        s = str_tensor_proto_type()
        self.assertIsInstance(s, str)
        self.assertIn("FLOAT", s)
        self.assertIn("INT64", s)

    def test_enumerate_subgraphs_simple(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "test",
                [_mkv_("X", TFLOAT, [3, 4])],
                [_mkv_("Y", TFLOAT, [3, 4])],
            )
        )
        graphs = list(enumerate_subgraphs(model.graph))
        self.assertEqual(len(graphs), 1)
        self.assertIs(graphs[0], model.graph)

    def test_overwrite_shape_in_model_proto(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "test",
                [_mkv_("X", TFLOAT, [3, 4])],
                [_mkv_("Y", TFLOAT, [3, 4])],
            )
        )
        new_model = overwrite_shape_in_model_proto(model)
        self.assertIsInstance(new_model, onnx.ModelProto)
        inp = new_model.graph.input[0]
        # After rewrite, shape dimensions should be symbolic (dim_param)
        for dim in inp.type.tensor_type.shape.dim:
            self.assertTrue(dim.dim_param or dim.dim_value == 0)

    def test_overwrite_shape_in_model_proto_n_in(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "test",
                [_mkv_("X", TFLOAT, [3, 4]), _mkv_("Y", TFLOAT, [3, 4])],
                [_mkv_("Z", TFLOAT, [3, 4])],
            )
        )
        new_model = overwrite_shape_in_model_proto(model, n_in=1)
        self.assertIsInstance(new_model, onnx.ModelProto)
        # First input should be rewritten, second should be unchanged
        inp0 = new_model.graph.input[0]
        for dim in inp0.type.tensor_type.shape.dim:
            self.assertTrue(dim.dim_param or dim.dim_value == 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
