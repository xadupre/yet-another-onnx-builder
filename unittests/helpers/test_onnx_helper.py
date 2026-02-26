import unittest
import ml_dtypes
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.helpers._onnx_simple_text_plot import onnx_simple_text_plot
from yobx.helpers.onnx_helper import onnx_dtype_name, tensor_dtype_to_np_dtype, pretty_onnx

TFLOAT = onnx.TensorProto.FLOAT


class TestOnnxHelper(ExtTestCase):
    def test_onnx_dtype_name(self):
        for k in dir(onnx.TensorProto):
            if k.upper() == k and k not in {"DESCRIPTOR", "EXTERNAL", "DEFAULT"}:
                self.assertEqual(k, onnx_dtype_name(getattr(onnx.TensorProto, k)))
        self.assertRaise(lambda: onnx_dtype_name(1000), ValueError)
        self.assertEqual(onnx_dtype_name(1000, exc=False), "UNEXPECTED")

    def test_tensor_dtype_to_np_dtype_standard(self):
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT), np.float32)
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.DOUBLE), np.float64)
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.INT32), np.int32)
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.INT64), np.int64)
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.BOOL), np.bool_)

    def test_tensor_dtype_to_np_dtype_float8(self):
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.BFLOAT16), ml_dtypes.bfloat16)
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E4M3FN),
            ml_dtypes.float8_e4m3fn,
        )
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E4M3FNUZ),
            ml_dtypes.float8_e4m3fnuz,
        )
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E5M2), ml_dtypes.float8_e5m2
        )
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E5M2FNUZ),
            ml_dtypes.float8_e5m2fnuz,
        )

    def test_pretty_onnx(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["added"]),
                    oh.make_node("Concat", ["added", "X"], ["concat_out"], axis=2),
                    oh.make_node("Reshape", ["concat_out", "reshape_shape"], ["Z"]),
                ],
                "add_concat_reshape",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", "d_model"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
                [
                    onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="reshape_shape"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        text = pretty_onnx(model)
        self.assertIn("Reshape(concat_out, reshape_shape) -> Z", text)

    def test_pretty_onnx_value_info_proto(self):
        vi = oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq"])
        text = pretty_onnx(vi)
        self.assertIn("FLOAT", text)
        self.assertIn("batch", text)
        self.assertIn("X", text)

    def test_pretty_onnx_type_proto(self):
        tp = oh.make_tensor_type_proto(TFLOAT, [3, 4])
        text = pretty_onnx(tp)
        self.assertIn("FLOAT", text)
        self.assertIn("3", text)

    def test_pretty_onnx_attribute_proto_int(self):
        att = oh.make_attribute("axis", 2)
        text = pretty_onnx(att)
        self.assertIn("axis=2", text)

    def test_pretty_onnx_attribute_proto_ints(self):
        att = oh.make_attribute("axes", [0, 1])
        text = pretty_onnx(att)
        self.assertIn("axes=", text)

    def test_pretty_onnx_attribute_proto_float(self):
        att = oh.make_attribute("eps", 1e-5)
        text = pretty_onnx(att)
        self.assertIn("eps=", text)

    def test_pretty_onnx_attribute_proto_floats(self):
        att = oh.make_attribute("coefs", [0.1, 0.2])
        text = pretty_onnx(att)
        self.assertIn("coefs=", text)

    def test_pretty_onnx_attribute_proto_string(self):
        att = oh.make_attribute("mode", "linear")
        text = pretty_onnx(att)
        self.assertIn("mode=", text)

    def test_pretty_onnx_attribute_proto_tensor(self):
        att = oh.make_attribute(
            "value", onh.from_array(np.array([1.0, 2.0], dtype=np.float32))
        )
        text = pretty_onnx(att)
        self.assertIn("value=", text)
        self.assertIn("tensor(", text)

    def test_pretty_onnx_node_proto(self):
        node = oh.make_node("Add", ["X", "Y"], ["Z"])
        text = pretty_onnx(node)
        self.assertEqual("Add(X, Y) -> Z", text)

    def test_pretty_onnx_node_proto_with_domain(self):
        node = oh.make_node("MatMul", ["A", "B"], ["C"], domain="com.microsoft")
        text = pretty_onnx(node)
        self.assertEqual("com.microsoft.MatMul(A, B) -> C", text)

    def test_pretty_onnx_node_proto_with_attributes(self):
        node = oh.make_node("Concat", ["X", "Y"], ["Z"], axis=1)
        text = pretty_onnx(node, with_attributes=True)
        self.assertIn("Concat(X, Y) -> Z", text)
        self.assertIn("axis=1", text)

    def test_pretty_onnx_tensor_proto(self):
        tensor = onh.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="W")
        text = pretty_onnx(tensor)
        self.assertIn("TensorProto", text)
        self.assertIn("2x2", text)
        self.assertIn("W", text)

    def test_pretty_onnx_sparse_tensor_proto(self):
        sparse = onnx.SparseTensorProto()
        self.assertRaise(lambda: pretty_onnx(sparse), AssertionError)

    def test_pretty_onnx_function_proto(self):
        func = oh.make_function(
            domain="test",
            fname="AddSub",
            inputs=["x", "y"],
            outputs=["s", "d"],
            nodes=[
                oh.make_node("Add", ["x", "y"], ["s"]),
                oh.make_node("Sub", ["x", "y"], ["d"]),
            ],
            opset_imports=[oh.make_opsetid("", 18)],
        )
        text = pretty_onnx(func)
        self.assertIn("function", text)
        self.assertIn("AddSub", text)

    def test_pretty_onnx_graph_proto(self):
        graph = oh.make_graph(
            [oh.make_node("Add", ["X", "Y"], ["Z"])],
            "test_graph",
            [
                oh.make_tensor_value_info("X", TFLOAT, [3, 4]),
                oh.make_tensor_value_info("Y", TFLOAT, [3, 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [3, 4])],
        )
        text = pretty_onnx(graph)
        self.assertIn("Add", text)
    def test_onnx_simple_text_plot_add_links(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["added"]),
                    oh.make_node("Concat", ["added", "X"], ["concat_out"], axis=2),
                    oh.make_node("Reshape", ["concat_out", "reshape_shape"], ["Z"]),
                ],
                "add_concat_reshape",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", "d_model"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
                [
                    onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="reshape_shape"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        text = onnx_simple_text_plot(model, add_links=True)
        self.assertIn("Reshape(concat_out, reshape_shape) -> Z", text)
        # add_links=True renders ASCII art links; intermediate link lines end with '|'
        self.assertTrue(any(line.endswith("|") for line in text.splitlines()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
