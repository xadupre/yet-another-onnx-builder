import unittest
import onnx
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape import BasicShapeBuilder, InferenceMode
from yobx.xshape.type_inference import infer_types

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16
TINT64 = onnx.TensorProto.INT64
TBOOL = onnx.TensorProto.BOOL


class TestTypeInference(ExtTestCase):
    def test_infer_types_relu(self):
        node = oh.make_node("Relu", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_add(self):
        node = oh.make_node("Add", ["X", "Y"], ["Z"])
        result = infer_types(node, [TFLOAT, TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_matmul(self):
        node = oh.make_node("MatMul", ["X", "Y"], ["Z"])
        result = infer_types(node, [TFLOAT, TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_cast(self):
        node = oh.make_node("Cast", ["X"], ["Y"], to=TINT64)
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_cast_like(self):
        node = oh.make_node("CastLike", ["X", "target"], ["Y"])
        result = infer_types(node, [TFLOAT, TINT64])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_constant_int(self):
        node = oh.make_node("Constant", [], ["C"], value_int=5)
        result = infer_types(node, [])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_constant_float(self):
        node = oh.make_node("Constant", [], ["C"], value_float=3.14)
        result = infer_types(node, [])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_constant_of_shape_default(self):
        node = oh.make_node("ConstantOfShape", ["shape"], ["C"])
        result = infer_types(node, [TINT64])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_shape(self):
        node = oh.make_node("Shape", ["X"], ["S"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_size(self):
        node = oh.make_node("Size", ["X"], ["S"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_where(self):
        node = oh.make_node("Where", ["cond", "X", "Y"], ["Z"])
        result = infer_types(node, [TBOOL, TFLOAT, TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_split(self):
        node = oh.make_node("Split", ["X"], ["Y1", "Y2"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, [TFLOAT, TFLOAT])

    def test_infer_types_range(self):
        node = oh.make_node("Range", ["start", "end", "step"], ["Y"])
        result = infer_types(node, [TINT64, TINT64, TINT64])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_eye_like_default(self):
        node = oh.make_node("EyeLike", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_eye_like_with_dtype(self):
        node = oh.make_node("EyeLike", ["X"], ["Y"], dtype=TINT64)
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_unsupported_no_exc(self):
        node = oh.make_node("UnknownCustomOp", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT], exc=False)
        self.assertEqual(result, 0)

    def test_infer_types_unsupported_exc(self):
        node = oh.make_node("UnknownCustomOp", ["X"], ["Y"])
        self.assertRaise(lambda: infer_types(node, [TFLOAT], exc=True), RuntimeError)

    def test_infer_types_output_name(self):
        node = oh.make_node("Relu", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT], output_name="Y")
        self.assertEqual(result, TFLOAT)

    def test_infer_types_reshape(self):
        node = oh.make_node("Reshape", ["X", "shape"], ["Y"])
        result = infer_types(node, [TFLOAT, TINT64])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_transpose(self):
        node = oh.make_node("Transpose", ["X"], ["Y"], perm=[1, 0])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_function_single_node(self):
        node = oh.make_node("Relu", ["X"], ["Y"])
        func = oh.make_function(
            domain="test",
            fname="MyRelu",
            inputs=["X"],
            outputs=["Y"],
            nodes=[node],
            opset_imports=[oh.make_opsetid("", 18)],
        )
        result = infer_types(func, [TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_function_multi_node(self):
        relu_node = oh.make_node("Relu", ["X"], ["T"])
        cast_node = oh.make_node("Cast", ["T"], ["Y"], to=TINT64)
        func = oh.make_function(
            domain="test",
            fname="ReluCast",
            inputs=["X"],
            outputs=["Y"],
            nodes=[relu_node, cast_node],
            opset_imports=[oh.make_opsetid("", 18)],
        )
        result = infer_types(func, [TFLOAT])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_function_multiple_outputs(self):
        split_node = oh.make_node("Split", ["X"], ["Y1", "Y2"])
        func = oh.make_function(
            domain="test",
            fname="MySplit",
            inputs=["X"],
            outputs=["Y1", "Y2"],
            nodes=[split_node],
            opset_imports=[oh.make_opsetid("", 18)],
        )
        result = infer_types(func, [TFLOAT])
        self.assertEqual(result, (TFLOAT, TFLOAT))

    def test_infer_types_floor(self):
        node = oh.make_node("Floor", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_trunc(self):
        node = oh.make_node("Trunc", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_dropout_single_output(self):
        node = oh.make_node("Dropout", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, [TFLOAT])

    def test_infer_types_dropout_with_mask(self):
        node = oh.make_node("Dropout", ["X"], ["Y", "mask"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, [TFLOAT, TBOOL])

    def test_infer_types_loop_loop_carried(self):
        body = oh.make_graph(
            [oh.make_node("Add", ["v", "v"], ["v_out"])],
            "loop_body",
            [
                oh.make_tensor_value_info("iter", TINT64, []),
                oh.make_tensor_value_info("cond_in", TBOOL, []),
                oh.make_tensor_value_info("v", TFLOAT, [3, 4]),
            ],
            [
                oh.make_tensor_value_info("cond_out", TBOOL, []),
                oh.make_tensor_value_info("v_out", TFLOAT, [3, 4]),
            ],
        )
        node = oh.make_node(
            "Loop",
            inputs=["max_iter", "cond", "v_in"],
            outputs=["v_final"],
            body=body,
        )
        result = infer_types(node, [TINT64, TBOOL, TFLOAT])
        self.assertEqual(list(result), [TFLOAT])

    def test_infer_types_loop_with_scan_output(self):
        body = oh.make_graph(
            [
                oh.make_node("Add", ["v", "v"], ["v_out"]),
                oh.make_node("Identity", ["v"], ["scan_out"]),
            ],
            "loop_body",
            [
                oh.make_tensor_value_info("iter", TINT64, []),
                oh.make_tensor_value_info("cond_in", TBOOL, []),
                oh.make_tensor_value_info("v", TFLOAT, [3, 4]),
            ],
            [
                oh.make_tensor_value_info("cond_out", TBOOL, []),
                oh.make_tensor_value_info("v_out", TFLOAT, [3, 4]),
                oh.make_tensor_value_info("scan_out", TFLOAT, [3, 4]),
            ],
        )
        node = oh.make_node(
            "Loop",
            inputs=["max_iter", "cond", "v_in"],
            outputs=["v_final", "scan"],
            body=body,
        )
        result = infer_types(node, [TINT64, TBOOL, TFLOAT])
        self.assertEqual(list(result), [TFLOAT, TFLOAT])

    def test_infer_types_loop_no_body(self):
        node = oh.make_node(
            "Loop",
            inputs=["max_iter", "cond", "v_in"],
            outputs=["v_final"],
        )
        result = infer_types(node, [TINT64, TBOOL, TFLOAT], exc=False)
        self.assertEqual(list(result), [0])

    def test_infer_types_loop_missing_body_output_types_inferred(self):
        """When body output types are not declared (0), they should be inferred
        by propagating types through the body graph nodes."""
        # Use make_tensor_value_info with no type (0) for body outputs
        body = oh.make_graph(
            [
                oh.make_node("Add", ["v", "v"], ["v_out"]),
                oh.make_node("Identity", ["v"], ["scan_out"]),
            ],
            "loop_body",
            [
                oh.make_tensor_value_info("iter", TINT64, []),
                oh.make_tensor_value_info("cond_in", TBOOL, []),
                # v input has no declared type - will be supplied via input_types
                oh.make_tensor_value_info("v", 0, None),
            ],
            [
                oh.make_tensor_value_info("cond_out", TBOOL, []),
                # output types are intentionally undeclared (0)
                oh.make_tensor_value_info("v_out", 0, None),
                oh.make_tensor_value_info("scan_out", 0, None),
            ],
        )
        node = oh.make_node(
            "Loop",
            inputs=["max_iter", "cond", "v_in"],
            outputs=["v_final", "scan"],
            body=body,
        )
        result = infer_types(node, [TINT64, TBOOL, TFLOAT])
        self.assertEqual(list(result), [TFLOAT, TFLOAT])


_mkv_ = oh.make_tensor_value_info


class TestRunModelTypeInference(ExtTestCase):
    def _make_model(self, nodes, inputs, outputs):
        return oh.make_model(
            oh.make_graph(nodes, "g", inputs, outputs),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )

    def test_run_model_type_only_relu(self):
        model = self._make_model(
            [oh.make_node("Relu", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model, inference=InferenceMode.TYPE)
        self.assertEqual(b.get_type("X"), TFLOAT)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertFalse(b.has_shape("Y"))

    def test_run_model_type_only_cast(self):
        model = self._make_model(
            [oh.make_node("Cast", ["X"], ["Y"], to=TINT64)],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TINT64, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model, inference=InferenceMode.TYPE)
        self.assertEqual(b.get_type("X"), TFLOAT)
        self.assertEqual(b.get_type("Y"), TINT64)

    def test_run_model_type_only_input_output_names(self):
        model = self._make_model(
            [oh.make_node("Relu", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model, inference=InferenceMode.TYPE)
        self.assertEqual(b.input_names, ["X"])
        self.assertEqual(b.output_names, ["Y"])

    def test_run_model_type_only_with_shape_inference(self):
        """Verify that SHAPE mode still works and infers shapes."""
        model = self._make_model(
            [oh.make_node("Relu", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model, inference=InferenceMode.SHAPE)
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertTrue(b.has_shape("Y"))

    def test_run_model_type_only_string_compat(self):
        """Verify that string values are still accepted for backward compat."""
        model = self._make_model(
            [oh.make_node("Relu", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        b.run_model(model, inference="type")
        self.assertEqual(b.get_type("Y"), TFLOAT)
        self.assertFalse(b.has_shape("Y"))

    def test_run_model_type_only_with_function(self):
        relu_node = oh.make_node("Relu", ["X"], ["Y"])
        func = oh.make_function(
            domain="test",
            fname="MyRelu",
            inputs=["X"],
            outputs=["Y"],
            nodes=[relu_node],
            opset_imports=[oh.make_opsetid("", 18)],
        )
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MyRelu", ["A"], ["B"], domain="test")],
                "g",
                [_mkv_("A", TFLOAT, [3, 4])],
                [_mkv_("B", TFLOAT, [3, 4])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("test", 1)],
            ir_version=10,
            functions=[func],
        )
        b = BasicShapeBuilder()
        b.run_model(model, inference=InferenceMode.TYPE)
        self.assertEqual(b.get_type("A"), TFLOAT)
        self.assertEqual(b.get_type("B"), TFLOAT)

    def test_run_model_invalid_inference_mode(self):
        model = self._make_model(
            [oh.make_node("Relu", ["X"], ["Y"])],
            [_mkv_("X", TFLOAT, [3, 4])],
            [_mkv_("Y", TFLOAT, [3, 4])],
        )
        b = BasicShapeBuilder()
        self.assertRaise(lambda: b.run_model(model, inference="unknown"), ValueError)


if __name__ == "__main__":
    unittest.main(verbosity=2)
