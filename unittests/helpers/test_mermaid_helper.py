import unittest
import onnx
import onnx.helper as oh
import numpy as np
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.helpers.mermaid_helper import to_mermaid


class TestMermaidHelper(ExtTestCase):
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
        self.assertIn("flowchart TD", mermaid)
        self.assertIn(":::input", mermaid)
        self.assertIn(":::output", mermaid)
        self.assertIn(":::op", mermaid)
        self.assertIn("LayerNormalization_", mermaid)
        self.assertIn("Add_", mermaid)
        self.assertIn("-->", mermaid)

    def test_graph_with_initializer(self):
        TFLOAT = onnx.TensorProto.FLOAT
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
        self.assertIn("flowchart TD", mermaid)
        self.assertIn(":::init", mermaid)
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
        self.assertIn("flowchart TD", mermaid)
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
        self.assertIn("-.->", mermaid)

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
        self.assertIn("classDef input", mermaid)
        self.assertIn("classDef init", mermaid)
        self.assertIn("classDef op", mermaid)
        self.assertIn("classDef output", mermaid)

    def test_edge_labels_with_shapes(self):
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
        # Shape inference should produce edge labels with dtype and shape info
        self.assertIn("FLOAT", mermaid)

    def test_scan_node(self):
        TFLOAT = onnx.TensorProto.FLOAT
        body_z = oh.make_tensor_value_info("body_z", TFLOAT, [])
        body_out = oh.make_tensor_value_info("body_out", TFLOAT, [])
        body_graph = oh.make_graph(
            [
                oh.make_node("Add", ["state", "x_elem"], ["sum_"]),
                oh.make_node("Mul", ["sum_", "scale"], ["body_z"]),
                oh.make_node("Identity", ["body_z"], ["body_out"]),
            ],
            "scan_body",
            [
                oh.make_tensor_value_info("state", TFLOAT, []),
                oh.make_tensor_value_info("x_elem", TFLOAT, []),
            ],
            [body_z, body_out],
        )
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Scan",
                        ["init", "X"],
                        ["final", "output"],
                        num_scan_inputs=1,
                        body=body_graph,
                    )
                ],
                "scan_graph",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [5]),
                    oh.make_tensor_value_info("init", TFLOAT, []),
                    oh.make_tensor_value_info("scale", TFLOAT, []),
                ],
                [
                    oh.make_tensor_value_info("final", TFLOAT, []),
                    oh.make_tensor_value_info("output", TFLOAT, [5]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        mermaid = to_mermaid(model)
        self.assertIn("Scan_", mermaid)
        self.assertIn("-.->", mermaid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
