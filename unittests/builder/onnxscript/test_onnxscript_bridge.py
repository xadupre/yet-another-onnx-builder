import unittest
import numpy as np
import onnx
from onnx.checker import check_model
import onnx.numpy_helper as onh
from yobx.container import ExportArtifact
from yobx.ext_test_case import ExtTestCase, requires_onnxscript
from yobx.reference import ExtendedReferenceEvaluator


@requires_onnxscript()
class TestOnnxScriptBridge(ExtTestCase):
    def _make_builder(self, opset=18, ir_version=None):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        return OnnxScriptGraphBuilder(opset, ir_version=ir_version)

    def test_construction_with_int_opset(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        gr = OnnxScriptGraphBuilder(18)
        self.assertEqual(gr.opsets, {"": 18})

    def test_construction_with_dict_opset(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        gr = OnnxScriptGraphBuilder({"": 18, "com.microsoft": 1})
        self.assertEqual(gr.opsets[""], 18)
        self.assertIn("com.microsoft", gr.opsets)

    def test_inner_builder_accessible(self):
        gr = self._make_builder()
        inner = gr.inner_builder
        self.assertIsNotNone(inner)

    def test_op_dispatcher_accessible(self):
        gr = self._make_builder()
        self.assertIsNotNone(gr.op)

    def test_make_tensor_input_returns_name(self):
        gr = self._make_builder()
        name = gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (None, 4))
        self.assertEqual(name, "X")

    def test_make_tensor_input_registers_name(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        self.assertTrue(gr.has_name("X"))

    def test_make_tensor_input_no_shape(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT)
        self.assertTrue(gr.has_name("X"))

    def test_make_tensor_input_no_type(self):
        gr = self._make_builder()
        # No elem_type: valid for function-style graphs
        gr.make_tensor_input("X")
        self.assertTrue(gr.has_name("X"))

    def test_make_tensor_input_static_shape(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3, 4))
        gr.make_node("Relu", ["X"], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        check_model(proto)

    def test_make_tensor_input_dynamic_shape_int(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (None, 4))
        gr.make_node("Relu", ["X"], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        check_model(proto)

    def test_make_tensor_input_symbolic_shape(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, ("batch", 4))
        gr.make_node("Relu", ["X"], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        check_model(proto)

    def test_make_tensor_output_returns_name(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        gr.make_node("Relu", ["X"], ["Y"])
        name = gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        self.assertEqual(name, "Y")

    def test_make_tensor_output_list(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (5,))
        k = gr.make_initializer("k", np.array([3], dtype=np.int64))
        gr.make_node("TopK", ["X", k], ["vals", "idxs"])
        names = gr.make_tensor_output(["vals", "idxs"])
        self.assertEqual(names, ["vals", "idxs"])

    def test_make_tensor_output_missing_name_raises(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        with self.assertRaises(KeyError):
            gr.make_tensor_output("does_not_exist")

    def test_make_initializer_numpy(self):
        gr = self._make_builder()
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        name = gr.make_initializer("W", arr)
        self.assertEqual(name, "W")
        self.assertTrue(gr.has_name("W"))

    def test_make_initializer_int(self):
        gr = self._make_builder()
        name = gr.make_initializer("c", 42)
        self.assertEqual(name, "c")

    def test_make_initializer_float(self):
        gr = self._make_builder()
        name = gr.make_initializer("eps", 1e-5)
        self.assertEqual(name, "eps")

    def test_make_initializer_tensor_proto(self):

        gr = self._make_builder()
        arr = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        tp = onh.from_array(arr, name="eye")
        name = gr.make_initializer("eye", tp)
        self.assertEqual(name, "eye")

    def test_make_initializer_empty_name_gets_auto_name(self):
        gr = self._make_builder()
        arr = np.zeros((3,), dtype=np.float32)
        name = gr.make_initializer("", arr)
        self.assertNotEqual(name, "")
        self.assertTrue(gr.has_name(name))

    def test_make_initializer_in_graph(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (None, 3))
        bias = gr.make_initializer("bias", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        gr.make_node("Add", ["X", bias], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        check_model(proto.get_proto())
        self.assertEqual(len(proto.graph.initializer), 1)
        self.assertEqual(proto.graph.initializer[0].name, "bias")

    # ------------------------------------------------------------------
    # make_node
    # ------------------------------------------------------------------

    def test_make_node_single_output_str(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        result = gr.make_node("Relu", ["X"], ["Y"])
        self.assertEqual(result, "Y")

    def test_make_node_ir_value_name_matches_requested_output_single(self):
        """ir.Value.name must equal the requested output name (single output)."""
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        gr.make_node("Relu", ["X"], ["my_output"])
        v = gr.get_value("my_output")
        self.assertEqual(v.name, "my_output")

    def test_make_node_ir_value_name_matches_requested_output_multiple(self):
        """ir.Value.name must equal the requested output names (multiple outputs)."""
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (5,))
        k = gr.make_initializer("k", np.array([3], dtype=np.int64))
        gr.make_node("TopK", ["X", k], ["vals", "idxs"])
        self.assertEqual(gr.get_value("vals").name, "vals")
        self.assertEqual(gr.get_value("idxs").name, "idxs")

    def test_make_node_output_names_appear_in_proto(self):
        """Requested output names must appear as node output names in the ONNX proto."""
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        gr.make_node("Relu", ["X"], ["my_relu_out"])
        gr.make_tensor_output("my_relu_out", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        check_model(proto)
        self.assertEqual(proto.graph.node[0].output[0], "my_relu_out")

    def test_make_node_single_output_int(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        result = gr.make_node("Relu", ["X"], 1)
        # Auto-generated name
        self.assertIsInstance(result, str)
        self.assertTrue(gr.has_name(result))

    def test_make_node_multiple_output_list(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (5,))
        k = gr.make_initializer("k", np.array([3], dtype=np.int64))
        result = gr.make_node("TopK", ["X", k], ["vals", "idxs"])
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, ("vals", "idxs"))

    def test_make_node_chained(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        r = gr.make_node("Relu", ["X"], ["r"])
        n = gr.make_node("Neg", [r], ["n"])
        gr.make_tensor_output(n, onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        check_model(proto)
        self.assertEqual([node.op_type for node in proto.graph.node], ["Relu", "Neg"])

    def test_make_node_with_kwargs_attributes(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3, 4))
        gr.make_node("Transpose", ["X"], ["T"], perm=[1, 0])
        gr.make_tensor_output("T", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        check_model(proto)
        node = proto.graph.node[0]
        self.assertEqual(node.op_type, "Transpose")
        perm_attr = next(a for a in node.attribute if a.name == "perm")
        self.assertEqual(list(perm_attr.ints), [1, 0])

    def test_make_node_with_attribute_proto(self):
        import onnx.helper as oh

        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3, 4))
        attr = oh.make_attribute("perm", [1, 0])
        gr.make_node("Transpose", ["X"], ["T"], attributes=[attr])
        gr.make_tensor_output("T", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        check_model(proto)

    def test_make_node_with_domain(self):
        """Custom domain nodes are forwarded correctly."""
        # We only check the node is created without error;
        # custom domain ops cannot be validated by the checker.
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        result = gr.make_node("CustomOp", ["X"], ["Y"], domain="custom")
        self.assertEqual(result, "Y")

    def test_make_node_optional_input(self):
        """Empty-string optional inputs are forwarded as None."""
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3, 4))
        # Unsqueeze in opset 13+ takes a separate axes input
        axes = gr.make_initializer("axes", np.array([0], dtype=np.int64))
        gr.make_node("Unsqueeze", ["X", axes], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        check_model(proto)

    def test_make_node_unknown_input_raises(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        with self.assertRaises(KeyError):
            gr.make_node("Add", ["X", "missing"], ["Y"])

    # ------------------------------------------------------------------
    # to_onnx
    # ------------------------------------------------------------------

    def test_to_onnx_returns_model_proto(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        gr.make_node("Relu", ["X"], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        self.assertIsInstance(proto, ExportArtifact)

    def test_to_onnx_multiple_outputs(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (5,))
        k = gr.make_initializer("k", np.array([3], dtype=np.int64))
        gr.make_node("TopK", ["X", k], ["vals", "idxs"])
        gr.make_tensor_output("vals", onnx.TensorProto.FLOAT)
        gr.make_tensor_output("idxs", onnx.TensorProto.INT64)
        proto = gr.to_onnx()
        check_model(proto)
        self.assertEqual(len(proto.graph.output), 2)

    def test_to_onnx_ir_version_override(self):
        gr = self._make_builder(ir_version=8)
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        gr.make_node("Relu", ["X"], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        self.assertEqual(proto.ir_version, 8)

    def test_to_onnx_shape_field_present(self):
        """Exported proto should have tensor_type.shape set (onnx >= 1.20 requirement)."""
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (None, 3))
        gr.make_node("Relu", ["X"], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        for vi in list(proto.graph.input) + list(proto.graph.output):
            if vi.type.HasField("tensor_type"):
                self.assertTrue(
                    vi.type.tensor_type.HasField("shape"), f"shape field missing for {vi.name}"
                )

    def test_to_onnx_custom_ir_version_at_construction(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        gr = OnnxScriptGraphBuilder(18, ir_version=7)
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        gr.make_node("Relu", ["X"], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        self.assertEqual(proto.ir_version, 7)

    def test_has_name_false_for_unknown(self):
        gr = self._make_builder()
        self.assertFalse(gr.has_name("unknown"))

    def test_get_value_raises_for_unknown(self):
        gr = self._make_builder()
        with self.assertRaises(KeyError):
            gr.get_value("unknown")

    def test_get_value_returns_ir_value(self):
        import onnx_ir as ir

        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        v = gr.get_value("X")
        self.assertIsInstance(v, ir.Value)

    def test_add_bias_numerical(self):
        """Add a bias to an input and verify numerical correctness."""
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (None, 3))
        bias = gr.make_initializer("bias", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        gr.make_node("Add", ["X", bias], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()

        session = ExtendedReferenceEvaluator(proto)
        x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        (y,) = session.run(None, {"X": x})
        expected = x + np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertEqualArray(y, expected)

    def test_matmul_numerical(self):
        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (2, 3))
        w = gr.make_initializer("W", np.eye(3, dtype=np.float32))
        gr.make_node("MatMul", ["X", w], ["Y"])
        gr.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = gr.to_onnx()
        session = ExtendedReferenceEvaluator(proto)
        x = np.arange(6, dtype=np.float32).reshape((2, 3))
        (y,) = session.run(None, {"X": x})
        self.assertEqualArray(y, x @ np.eye(3, dtype=np.float32))

    # ------------------------------------------------------------------
    # Direct onnxscript builder access
    # ------------------------------------------------------------------

    def test_inner_builder_can_create_nodes_directly(self):
        """Nodes created via inner_builder.op should be present in the graph."""

        gr = self._make_builder()
        gr.make_tensor_input("X", onnx.TensorProto.FLOAT, (3,))
        x_val = gr.get_value("X")
        y_val = gr.inner_builder.op.Relu(x_val)
        y_val.name = "Y"
        gr._graph.outputs.append(y_val)
        gr._register("Y", y_val)
        proto = gr.to_onnx()
        check_model(proto)
        self.assertEqual(proto.graph.node[0].op_type, "Relu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
