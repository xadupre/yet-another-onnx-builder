import unittest
from yobx._onnx_shim import onnx
import onnx_light.onnx.helper as oh
from yobx.ext_test_case import ExtTestCase
from onnx_light.tools import to_mermaid


class TestMermaidEmitter(ExtTestCase):
    def test_basic_graph(self):
        TFLOAT16 = onnx.TensorProto.FLOAT16
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "LayerNormalization",
                        ["X", "W", "B"],
                        ["ln"],
                        axis=-1,
                        epsilon=9.999999974752427e-7,
                    ),
                    oh.make_node("Add", ["ln", "W"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT16, ["b", "c", "d"]),
                    oh.make_tensor_value_info("W", TFLOAT16, ["d"]),
                    oh.make_tensor_value_info("B", TFLOAT16, ["d"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT16, ["b", "c", "d"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        mermaid = to_mermaid(model)
        self.assertIn("flowchart TB", mermaid)
        self.assertIn(":::onnxInput", mermaid)
        self.assertIn(":::onnxOutput", mermaid)
        self.assertIn(":::onnxOp", mermaid)
        self.assertIn("LayerNormalization_", mermaid)
        self.assertIn("Add_", mermaid)
        self.assertIn("-->", mermaid)

    def test_graph_with_initializer(self):
        TFLOAT = onnx.TensorProto.FLOAT
        import numpy as np
        import onnx_light.onnx.numpy_helper as onh

        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MatMul", ["X", "W"], ["mm"]), oh.make_node("Relu", ["mm"], ["Z"])],
                "matmul_relu",
                [oh.make_tensor_value_info("X", TFLOAT, [4, 4])],
                [oh.make_tensor_value_info("Z", TFLOAT, [4, 2])],
                [onh.from_array(np.zeros((4, 2), dtype=np.float32), name="W")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        mermaid = to_mermaid(model)
        self.assertIn("flowchart TB", mermaid)
        self.assertIn(":::onnxInitializer", mermaid)
        self.assertIn("MatMul_", mermaid)
        self.assertIn("Relu_", mermaid)

    def test_graph_with_constant_node(self):
        TFLOAT16 = onnx.TensorProto.FLOAT16
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "LayerNormalization",
                        ["X", "W", "B"],
                        ["ln"],
                        axis=-1,
                        epsilon=9.999999974752427e-7,
                    ),
                    oh.make_node("Constant", [], ["cst"], value_float=[1]),
                    oh.make_node("Cast", ["cst"], ["cst16"], to=onnx.TensorProto.FLOAT16),
                    oh.make_node("Add", ["ln", "cst16"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT16, ["b", "c", "d"]),
                    oh.make_tensor_value_info("W", TFLOAT16, ["d"]),
                    oh.make_tensor_value_info("B", TFLOAT16, ["d"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT16, ["b", "c", "d"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        mermaid = to_mermaid(model)
        self.assertIn("flowchart TB", mermaid)
        self.assertIn("Cast_", mermaid)
        self.assertIn("Add_", mermaid)

    def test_if_node(self):
        TFLOAT = onnx.TensorProto.FLOAT
        then_z = oh.make_tensor_value_info("then_z", TFLOAT, [3])
        then_graph = oh.make_graph(
            [oh.make_node("Add", ["X", "X"], ["then_z"])], "then_branch", [], [then_z]
        )
        else_z = oh.make_tensor_value_info("else_z", TFLOAT, [3])
        else_graph = oh.make_graph(
            [oh.make_node("Neg", ["X"], ["else_z"])], "else_branch", [], [else_z]
        )
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "If", ["cond"], ["Z"], then_branch=then_graph, else_branch=else_graph
                    )
                ],
                "if_graph",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [3]),
                    oh.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        mermaid = to_mermaid(model)
        self.assertIn("If_", mermaid)
        self.assertIn(":::onnxOp", mermaid)

    def test_classdefs_present(self):
        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "relu_graph",
                [oh.make_tensor_value_info("X", TFLOAT, [3])],
                [oh.make_tensor_value_info("Y", TFLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        mermaid = to_mermaid(model)
        self.assertIn("classDef onnxInput", mermaid)
        self.assertIn("classDef onnxInitializer", mermaid)
        self.assertIn("classDef onnxOp", mermaid)
        self.assertIn("classDef onnxOutput", mermaid)
        # Each classDef must include explicit stroke and text colour so that
        # node borders and labels are visible in both light and dark themes.
        for cls in ("onnxInput", "onnxInitializer", "onnxOp", "onnxOutput"):
            self.assertIn(f"classDef {cls} fill:#", mermaid)
            # stroke and color must be present in the same classDef line
            lines = [line for line in mermaid.splitlines() if f"classDef {cls} " in line]
            self.assertEqual(len(lines), 1)
            self.assertIn("stroke:", lines[0])
            self.assertIn("color:", lines[0])

    def test_edge_labels(self):
        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "relu_graph",
                [oh.make_tensor_value_info("X", TFLOAT, [3])],
                [oh.make_tensor_value_info("Y", TFLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        mermaid = to_mermaid(model)
        # Edge labels should include dtype and shape info
        self.assertIn("float[3]", mermaid)

    def test_importable_from_onnx_light_tools(self):
        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "relu_graph",
                [oh.make_tensor_value_info("X", TFLOAT, [3])],
                [oh.make_tensor_value_info("Y", TFLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        mermaid = to_mermaid(model)
        self.assertIn("flowchart TB", mermaid)

    def test_mermaid_graph_directly(self):
        from onnx_light.tools import to_mermaid_graph

        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "relu_graph",
                [oh.make_tensor_value_info("X", TFLOAT, [3])],
                [oh.make_tensor_value_info("Y", TFLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        mermaid = to_mermaid_graph(model.graph)
        self.assertIn("flowchart TB", mermaid)
        self.assertIn("Relu_", mermaid)
        self.assertIn(":::onnxInput", mermaid)
        self.assertIn(":::onnxOutput", mermaid)

    def test_yobx_helper_reexports_native_renderer(self):
        from yobx.helpers.mermaid_helper import to_mermaid as yobx_to_mermaid

        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "relu_graph",
                [oh.make_tensor_value_info("X", TFLOAT, [3])],
                [oh.make_tensor_value_info("Y", TFLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self.assertEqual(yobx_to_mermaid(model), to_mermaid(model))


if __name__ == "__main__":
    unittest.main(verbosity=2)
