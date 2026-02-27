import textwrap
import unittest
import onnx
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase
from yobx.helpers.dot_helper import to_dot


class TestDotHelper(ExtTestCase):
    def test_custom_doc_kernels_layer_normalization(self):
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
                    oh.make_node(
                        "Add", ["ln", "W"], ["Z"], axis=-1, epsilon=9.999999974752427e-7
                    ),
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
        dot = to_dot(model)
        expected = textwrap.dedent("""
            digraph {
              graph [rankdir=TB, splines=true, overlap=false, nodesep=0.2, ranksep=0.2, fontsize=8];
              node [style="rounded,filled", color="#888888", fontcolor="#222222", shape=box];
              edge [arrowhead=vee, fontsize=7, labeldistance=-5, labelangle=0];
              I_0 [label="X\\nFLOAT16(b,c,d)", fillcolor="#aaeeaa"];
              I_1 [label="W\\nFLOAT16(d)", fillcolor="#aaeeaa"];
              I_2 [label="B\\nFLOAT16(d)", fillcolor="#aaeeaa"];
              LayerNormalization_3 [label="LayerNormalization(., ., ., axis=-1)", fillcolor="#cccccc"];
              Add_4 [label="Add(., ., axis=-1)", fillcolor="#cccccc"];
              I_0 -> LayerNormalization_3 [label="FLOAT16(b,c,d)"];
              I_1 -> LayerNormalization_3 [label="FLOAT16(d)"];
              I_2 -> LayerNormalization_3 [label="FLOAT16(d)"];
              LayerNormalization_3 -> Add_4 [label="FLOAT16(b,c,d)"];
              I_1 -> Add_4 [label="FLOAT16(d)"];
              O_5 [label="Z\\nFLOAT16(b,c,d)", fillcolor="#aaaaee"];
              Add_4 -> O_5;
            }
            """)
        self.maxDiff = None
        self.assertEqual(expected.strip("\n "), dot.strip("\n "))

    def test_custom_doc_kernels_layer_normalization_constant(self):
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
                    oh.make_node("Add", ["ln", "cst16"], ["Z"], axis=-1),
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
        dot = to_dot(model)
        expected = textwrap.dedent("""
                digraph {
                graph [rankdir=TB, splines=true, overlap=false, nodesep=0.2, ranksep=0.2, fontsize=8];
                node [style="rounded,filled", color="#888888", fontcolor="#222222", shape=box];
                edge [arrowhead=vee, fontsize=7, labeldistance=-5, labelangle=0];
                I_0 [label="X\\nFLOAT16(b,c,d)", fillcolor="#aaeeaa"];
                I_1 [label="W\\nFLOAT16(d)", fillcolor="#aaeeaa"];
                I_2 [label="B\\nFLOAT16(d)", fillcolor="#aaeeaa"];
                LayerNormalization_3 [label="LayerNormalization(., ., ., axis=-1)", fillcolor="#cccccc"];
                Cast_4 [label="Cast([1.0], to=FLOAT16)", fillcolor="#cccccc"];
                Add_5[label="Add(.,.,axis=-1)",fillcolor="#cccccc"];
                I_0 -> LayerNormalization_3 [label="FLOAT16(b,c,d)"];
                I_1 -> LayerNormalization_3 [label="FLOAT16(d)"];
                I_2 -> LayerNormalization_3 [label="FLOAT16(d)"];
                LayerNormalization_3 -> Add_5 [label="FLOAT16(b,c,d)"];
                Cast_4->Add_5[label="FLOAT16(1)"];
                O_6 [label="Z\\nFLOAT16(b,c,d)", fillcolor="#aaaaee"];
                Add_5 -> O_6;
                }
                """).strip("\n").replace(" ", "")
        self.maxDiff = None
        self.assertEqual(expected, dot.strip("\n").replace(" ", ""))


    def test_to_dot_if(self):
        TFLOAT = onnx.TensorProto.FLOAT
        then_z = oh.make_tensor_value_info("then_z", TFLOAT, [3])
        then_graph = oh.make_graph(
            [oh.make_node("Add", ["X", "X"], ["then_z"])],
            "then_branch",
            [],
            [then_z],
        )
        else_z = oh.make_tensor_value_info("else_z", TFLOAT, [3])
        else_graph = oh.make_graph(
            [oh.make_node("Neg", ["X"], ["else_z"])],
            "else_branch",
            [],
            [else_z],
        )
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "If",
                        ["cond"],
                        ["Z"],
                        then_branch=then_graph,
                        else_branch=else_graph,
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
        dot = to_dot(model)
        self.assertIn("If_", dot)
        self.assertIn("style=dotted", dot)

    def test_to_dot_scan(self):
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
        dot = to_dot(model)
        self.assertIn("Scan_", dot)
        self.assertIn("style=dotted", dot)

    def test_to_dot_loop(self):
        TFLOAT = onnx.TensorProto.FLOAT
        TINT64 = onnx.TensorProto.INT64
        TBOOL = onnx.TensorProto.BOOL
        body_cond = oh.make_tensor_value_info("cond_out", TBOOL, [])
        body_v = oh.make_tensor_value_info("v_out", TFLOAT, [])
        body_graph = oh.make_graph(
            [
                oh.make_node("Identity", ["cond_in"], ["cond_out"]),
                oh.make_node("Mul", ["v_in", "scale"], ["v_out"]),
            ],
            "loop_body",
            [
                oh.make_tensor_value_info("iter", TINT64, []),
                oh.make_tensor_value_info("cond_in", TBOOL, []),
                oh.make_tensor_value_info("v_in", TFLOAT, []),
            ],
            [body_cond, body_v],
        )
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Loop", ["max_iter", "cond", "v0"], ["v_final"], body=body_graph
                    )
                ],
                "loop_graph",
                [
                    oh.make_tensor_value_info("max_iter", TINT64, []),
                    oh.make_tensor_value_info("cond", TBOOL, []),
                    oh.make_tensor_value_info("v0", TFLOAT, []),
                    oh.make_tensor_value_info("scale", TFLOAT, []),
                ],
                [oh.make_tensor_value_info("v_final", TFLOAT, [])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        dot = to_dot(model)
        self.assertIn("Loop_", dot)
        self.assertIn("style=dotted", dot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
