import unittest
import onnx
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape.shape_builder_impl import BasicShapeBuilder
import yobx.xshape.basic_skill  # registers BasicSkill  # noqa: F401

TFLOAT = onnx.TensorProto.FLOAT
_mkv_ = oh.make_tensor_value_info


class TestBasicSkill(ExtTestCase):
    def test_basic_skill_preserves_static_shape(self):
        """Output of BasicSkill has the same static shape and type as its input."""
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("BasicSkill", ["X"], ["Y"], domain="custom")],
                "basic_skill_graph",
                [_mkv_("X", TFLOAT, [2, 4])],
                [_mkv_("Y", TFLOAT, [2, 4])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("custom", 1)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertTrue(builder.has_shape("Y"))
        self.assertEqual(builder.get_shape("Y"), (2, 4))
        self.assertEqual(builder.get_type("Y"), TFLOAT)

    def test_basic_skill_preserves_dynamic_shape(self):
        """Output of BasicSkill preserves dynamic (symbolic) dimensions."""
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("BasicSkill", ["X"], ["Y"], domain="custom")],
                "basic_skill_dynamic",
                [_mkv_("X", TFLOAT, ["batch", "seq", 16])],
                [_mkv_("Y", TFLOAT, ["batch", "seq", 16])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("custom", 1)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertTrue(builder.has_shape("Y"))
        self.assertEqual(builder.get_shape("Y"), ("batch", "seq", 16))
        self.assertEqual(builder.get_type("Y"), TFLOAT)

    def test_register_custom_shape_function(self):
        """Users can register their own shape inference for a custom operator."""
        from yobx.xshape.basic_skill import register_custom_shape_function
        from yobx.xshape.shape_type_compute import set_type_shape_unary_op

        register_custom_shape_function(
            "MyCustomOp",
            lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
        )

        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MyCustomOp", ["X"], ["Z"], domain="my_domain")],
                "my_graph",
                [_mkv_("X", TFLOAT, [4, 8])],
                [_mkv_("Z", TFLOAT, [4, 8])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("my_domain", 1)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(builder.get_shape("Z"), (4, 8))
        self.assertEqual(builder.get_type("Z"), TFLOAT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
