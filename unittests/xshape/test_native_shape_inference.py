import unittest

import numpy as np
from onnx_light import onnx
from onnx_light.onnx import helper, numpy_helper
from onnx_light.onnx_core.shape_inference import ShapesContext

from yobx.xshape import InferenceMode, NativeShapeInference
from yobx.xshape.cost_inference import estimate_node_flops
from yobx.xshape.shape_builder import ShapeBuilder


class TestNativeShapeInference(unittest.TestCase):
    @staticmethod
    def make_model(shape=("N", 3), op_type="Add", domain=""):
        """Creates a symbolic binary model with native protos."""
        return helper.make_model(
            helper.make_graph(
                [helper.make_node(op_type, ["X", "Y"], ["Z"], domain=domain)],
                "symbolic",
                [
                    helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape),
                    helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3]),
                ],
                [helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, None)],
            ),
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom", 1)],
        )

    def test_symbolic_add_and_cost(self):
        model = self.make_model()
        inference = NativeShapeInference()
        self.assertIsInstance(inference.context, ShapesContext)
        costs = inference.run_model(model, inference=InferenceMode.COST)
        self.assertEqual(inference.get_shape("Z"), ("N", 3))
        self.assertEqual(inference.get_type("Z"), onnx.TensorProto.FLOAT)
        self.assertEqual(inference.input_names, ["X", "Y"])
        self.assertEqual(inference.output_names, ["Z"])
        self.assertEqual(len(costs), 1)
        self.assertEqual(inference.estimate_node_flops(model.graph.node[0]), costs[0][1])
        evaluated = inference.evaluate_cost_with_true_inputs({"X": np.zeros((5, 3))}, costs)
        self.assertEqual(evaluated, [("Add", 15, (("N", 3), (1, 3)))])

    def test_serialization_preserves_source_and_metadata(self):
        model = self.make_model()
        model.producer_name = "test-producer"
        model.doc_string = "preserved"
        original = model.SerializeToString()
        inference = NativeShapeInference()
        inference.run_model(model)
        inferred = inference.to_onnx()
        self.assertEqual(model.SerializeToString(), original)
        self.assertEqual(inferred.producer_name, "test-producer")
        self.assertEqual(inferred.doc_string, "preserved")
        restored = onnx.ModelProto()
        restored.ParseFromString(inferred.SerializeToString())
        dims = restored.graph.output[0].type.tensor_type.shape.dim
        self.assertEqual(str(dims[0].dim_param), "N")
        self.assertEqual(dims[1].dim_value, 3)

    def test_native_incremental_inference(self):
        inference = NativeShapeInference()
        inference.set_type("X", onnx.TensorProto.FLOAT)
        inference.set_shape("X", ("N", 3))
        inference.set_type("Y", onnx.TensorProto.FLOAT)
        inference.set_shape("Y", (1, 3))
        inference.run_node(helper.make_node("Add", ["X", "Y"], ["Z"]), cost=False)
        self.assertEqual(inference.get_shape("Z"), ("N", 3))

    def test_native_shape_values_and_reshape(self):
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Shape", ["X"], ["shape"]),
                    helper.make_node("Reshape", ["X", "shape"], ["Y"]),
                    helper.make_node("Reshape", ["Y", "flat"], ["Z"]),
                ],
                "shape_values",
                [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["N", 3])],
                [helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, None)],
                [numpy_helper.from_array(np.array([-1], dtype=np.int64), name="flat")],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        inference = NativeShapeInference()
        inference.run_model(model)
        self.assertEqual(inference.value_as_shape("shape"), ("N", 3))
        self.assertEqual(inference.get_shape("Y"), ("N", 3))
        self.assertEqual(inference.get_rank("Z"), 1)
        self.assertIsNone(inference.value_as_shape("X"))
        self.assertIsNone(inference.value_as_shape("missing"))

    def test_native_unknown_dimensions(self):
        inference = NativeShapeInference()
        costs = inference.run_model(self.make_model((None, 3)), inference="cost")
        self.assertEqual(inference.get_shape("Z"), (None, 3))
        self.assertFalse(inference.has_shape("Z", full=True))
        self.assertIsNone(costs[0][1])

    def test_unknown_rank_is_not_a_scalar(self):
        with self.assertRaisesRegex(ValueError, "unknown rank"):
            NativeShapeInference().run_model(self.make_model(None))
        inference = NativeShapeInference()
        inference.set_type("X", onnx.TensorProto.FLOAT)
        self.assertFalse(inference.has_shape("X"))
        with self.assertRaisesRegex(ValueError, "unknown rank"):
            inference.run_node(helper.make_node("Identity", ["X"], ["Y"]))

    def test_native_constraints(self):
        inference = NativeShapeInference()
        inference.register_constraint_dimension("N", "M")
        self.assertTrue(inference.context.has_constraint("N", "M"))
        constraints = inference.get_registered_constraints()
        self.assertTrue("M" in constraints.get("N", set()) or "N" in constraints.get("M", set()))

    def test_shape_api_keyword_arguments(self):
        inference: ShapeBuilder = NativeShapeInference()
        inference.set_type(name="X", itype=onnx.TensorProto.FLOAT)
        inference.set_shape(name="X", shape=(None, "N/:2", 3))
        inference.set_device(name="X", device=2)
        inference.set_opset(name="", version=19)
        self.assertEqual(inference.get_type("X"), onnx.TensorProto.FLOAT)
        self.assertEqual(inference.get_device("X"), 2)
        self.assertEqual(inference.get_opset(""), 19)
        self.assertEqual(inference.evaluate_shape(name="X", context={"N": 8}), (None, 4, 3))
        with self.assertRaises(RuntimeError):
            inference.evaluate_shape(name="X", context={})

    def test_native_constraint_sets(self):
        inference: ShapeBuilder = NativeShapeInference()
        inference.register_constraint_dimension(dim_name="N", value={"M", 4})
        inference.add_to_constraints(dim_name="N", value="K")
        constraints = inference.get_registered_constraints()
        self.assertEqual(constraints["N"], {"M", "K", 4})
        self.assertEqual(constraints["M"], {"N"})

    def test_unknown_dimensions_in_cost_callbacks(self):
        node = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        shapes = {"X": (None, 3), "Y": (3, 4), "Z": (None, 4)}
        self.assertIsNone(estimate_node_flops(node, shapes.get, lambda name: None))
        literals = {"X": (2, None), "Y": (None, 4), "Z": (2, 4)}
        self.assertIsNone(estimate_node_flops(node, lambda name: None, literals.get))

    def test_rank_cost_preserves_unknown_dimensions(self):
        inference = NativeShapeInference()
        inference.set_type("X", onnx.TensorProto.FLOAT)
        inference.set_shape("X", (None, 3))
        node = helper.make_node("Shape", ["X"], ["shape"])
        self.assertEqual(inference.run_node(node), 2)
        self.assertEqual(inference.value_as_shape("shape"), (None, 3))

    def test_unknown_operator_does_not_fallback(self):
        for exc in (False, True):
            with self.subTest(exc=exc), self.assertRaises(ValueError):
                NativeShapeInference().run_model(
                    self.make_model(op_type="Unknown", domain="custom"), exc=exc
                )

    def test_custom_native_callback(self):
        inference = NativeShapeInference()
        calls = []

        def callback(context, node):
            calls.append(str(node.op_type))
            context.set(str(node.output[0]), context.get(str(node.input[0])))

        inference.context.set_custom_shape_inference_function("custom", "Unknown", callback)
        inference.run_model(self.make_model(op_type="Unknown", domain="custom"))
        self.assertEqual(calls, ["Unknown"])
        self.assertEqual(inference.get_shape("Z"), ("N", 3))

    def test_modes_and_reuse(self):
        inference = NativeShapeInference()
        inference.run_model(self.make_model(), inference="type")
        self.assertEqual(inference.get_type("Z"), onnx.TensorProto.FLOAT)
        self.assertFalse(inference.has_shape("Z"))
        inference.run_model(self.make_model(), inference="nothing")
        self.assertFalse(inference.has_type("Z"))
        inference.run_model(self.make_model((2, 3)))
        self.assertEqual(inference.get_shape("Z"), (2, 3))
        with self.assertRaises(ValueError):
            inference.run_model(self.make_model(), inference="invalid")

    def test_reset_for_independent_model(self):
        inference = NativeShapeInference()
        inference.run_model(self.make_model(), inference="type")
        inference.reset_types_and_shapes()
        self.assertEqual(inference.input_names, [])
        inference.run_model(self.make_model((4, 3)).graph)
        self.assertEqual(inference.get_shape("Z"), (4, 3))

    def test_unresolved_symbolic_cost(self):
        inference = NativeShapeInference()
        costs = inference.run_model(self.make_model(), inference="cost")
        self.assertIsNone(inference.evaluate_cost_with_true_inputs({}, costs)[0][1])
        with self.assertRaises(RuntimeError):
            inference.evaluate_cost_with_true_inputs({}, costs, exc=True)


if __name__ == "__main__":
    unittest.main()
