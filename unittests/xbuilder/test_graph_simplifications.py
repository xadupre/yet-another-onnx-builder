import unittest
import numpy as np
from yobx._onnx_shim import onnx
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
import onnx_light.onnx.checker as oc
from onnx_light.onnx import TensorProto
from yobx.ext_test_case import ExtTestCase
from yobx.xbuilder.graph_builder import GraphBuilder
from yobx.reference import ExtendedReferenceEvaluator


class TestGraphSimplification(ExtTestCase):
    def call_optimizer(self, onx):
        gr = GraphBuilder(onx)
        gr.inner_builder.remove_unused_nodes()
        gr.inner_builder.remove_identity_nodes()
        result = gr.to_onnx(optimize=True)
        self.assertEqual(
            [str(v.name) for v in onx.graph.output], [str(v.name) for v in result.graph.output]
        )
        return result

    def test_remove_unused_nodes(self):
        model = onnx.parser.parse_model("""
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(x, x)
            }""")
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 1)
        self.assertEqual(onx.graph.node[0].op_type, "Mul")

    def test_initializers(self):
        model = onnx.parser.parse_model("""
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z)
            <float two = {2.0}> {
                four = Add(two, two)
                z = Mul(x, x)
            }""")
        self.assertEqual(len(model.graph.initializer), 1)
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 1)
        self.assertEqual(onx.graph.node[0].op_type, "Mul")
        self.assertEqual(len(onx.graph.initializer), 0)

    def test_keep_unused_outputs(self):
        model = onnx.parser.parse_model("""
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[M] z) {
                w1, w2, w3 = Split (x)
                z = Mul(w3, w3)
            }""")
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 2)
        self.assertEqual(onx.graph.node[0].op_type, "Split")

    def test_remove_identity(self):
        opset_imports = [oh.make_opsetid("", 12)]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("input", TensorProto.FLOAT, shape=(2, 3)))
        nodes.append(oh.make_node("Softmax", ["input"], ["output_0"], axis=0))
        nodes.append(oh.make_node("Identity", ["output_0"], ["output_1"]))
        outputs.append(oh.make_tensor_value_info("output_0", TensorProto.FLOAT, shape=(2, 3)))
        outputs.append(oh.make_tensor_value_info("output_1", TensorProto.FLOAT, shape=(2, 3)))
        graph = oh.make_graph(
            nodes,
            "experiment",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)
        self.assertEqual(len(model.graph.node), 2)
        self.assertEqual(model.graph.node[0].op_type, "Softmax")
        self.assertEqual(model.graph.node[1].op_type, "Identity")
        self.assertEqual(len(model.graph.output), 2)
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 2)
        self.assertEqual(onx.graph.node[0].op_type, "Softmax")
        self.assertEqual(onx.graph.node[1].op_type, "Identity")
        self.assertEqual(len(model.graph.output), 2)

    def test_builder(self):
        gr = GraphBuilder(18, ir_version=9)
        gr.make_tensor_input("X", TensorProto.FLOAT, ("a", "b"))
        weight = gr.make_initializer("", np.array([[0.4, 0.5, 0.6]], dtype=np.float32).T)
        bias = gr.make_initializer("", np.array([[0.4, 0.5, 0.6]], dtype=np.float32))
        mm = gr.make_node("MatMul", ["X", weight], name="ut")
        out = gr.make_node("Add", [mm, bias], ["Y"], name="ut")
        gr.make_tensor_output(out, TensorProto.FLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()

        ref = ExtendedReferenceEvaluator(onx)
        x = np.random.rand(10, 3).astype(np.float32)
        y = ref.run(None, {"X": x})[0]
        self.assertEqual(y.dtype, np.float32)

    def test_builder_api2(self):
        gr = GraphBuilder(18, ir_version=9)
        gr.make_tensor_input("X", TensorProto.FLOAT, ("a", "b"))
        mm = gr.op.MatMul("X", np.array([[0.4, 0.5, 0.6]], dtype=np.float32).T)
        out = gr.op.Add(mm, np.array([0.4, 0.5, 0.6], dtype=np.float32), outputs=["Y"])
        gr.make_tensor_output(out, TensorProto.FLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()

        ref = ExtendedReferenceEvaluator(onx)
        x = np.random.rand(10, 3).astype(np.float32)
        y = ref.run(None, {"X": x})[0]
        self.assertEqual(y.dtype, np.float32)

    def test_remove_identity_two_paths1(self):
        opset_imports = [oh.make_opsetid("", 12)]
        nodes = [
            oh.make_node("Add", ["X", "Y"], ["add"]),
            oh.make_node("Identity", ["add"], ["add1"]),
            oh.make_node("Identity", ["add1"], ["add2"]),
            oh.make_node("Sub", ["add2", "X"], ["output1"]),
            oh.make_node("Identity", ["add"], ["output0"]),
        ]

        graph = oh.make_graph(
            nodes,
            "experiment",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5]),
            ],
            [
                oh.make_tensor_value_info("output0", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("output1", TensorProto.FLOAT, [4, 5]),
            ],
        )
        model = oh.make_model(graph, opset_imports=opset_imports)
        onx = self.call_optimizer(model)
        self.assertEqual(["Add", "Sub", "Identity"], [n.op_type for n in onx.graph.node])
        oc.check_model(onx.proto)

    def test_remove_identity_two_paths2(self):
        opset_imports = [oh.make_opsetid("", 12)]
        nodes = [
            oh.make_node("Add", ["X", "Y"], ["add"]),
            oh.make_node("Identity", ["add"], ["add1"]),
            oh.make_node("Identity", ["add1"], ["add2"]),
            oh.make_node("Identity", ["add"], ["output0"]),
            oh.make_node("Sub", ["add2", "X"], ["output1"]),
        ]

        graph = oh.make_graph(
            nodes,
            "experiment",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5]),
            ],
            [
                oh.make_tensor_value_info("output0", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("output1", TensorProto.FLOAT, [4, 5]),
            ],
        )
        model = oh.make_model(graph, opset_imports=opset_imports)
        onx = self.call_optimizer(model)
        self.assertEqual(["Add", "Identity", "Sub"], [n.op_type for n in onx.graph.node])
        oc.check_model(onx.proto)

    def test_remove_identity_two_paths3(self):
        opset_imports = [oh.make_opsetid("", 12)]
        nodes = [
            oh.make_node("Add", ["X", "Y"], ["add"]),
            oh.make_node("Identity", ["add"], ["add1"]),
            oh.make_node("Identity", ["add1"], ["output1"]),
            oh.make_node("Identity", ["add"], ["output0"]),
        ]

        graph = oh.make_graph(
            nodes,
            "experiment",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5]),
            ],
            [
                oh.make_tensor_value_info("output0", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("output1", TensorProto.FLOAT, [4, 5]),
            ],
        )
        model = oh.make_model(graph, opset_imports=opset_imports)
        onx = self.call_optimizer(model)
        self.assertEqual(["Add", "Identity", "Identity"], [n.op_type for n in onx.graph.node])
        oc.check_model(onx.proto)

    def test_remove_identity_shadowing(self):
        def _mkv_(name):
            value_info_proto = onnx.ValueInfoProto()
            value_info_proto.name = name
            return value_info_proto

        def _make_model(shadowing=True):
            local_name = "three" if shadowing else "local_three"
            return oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("ReduceSum", ["X"], ["Xred"]),
                        oh.make_node("Add", ["X", "two"], ["X0"]),
                        oh.make_node("Add", ["X0", "zero"], ["X00"]),
                        oh.make_node("CastLike", ["one", "Xred"], ["one_c"]),
                        oh.make_node("Greater", ["Xred", "one_c"], ["cond"]),
                        oh.make_node("Identity", ["two"], ["three"]),
                        oh.make_node(
                            "If",
                            ["cond"],
                            ["Z_c"],
                            then_branch=oh.make_graph(
                                [
                                    oh.make_node(
                                        "Constant", [], [local_name], value_floats=[2.1]
                                    ),
                                    oh.make_node("Add", ["X00", local_name], ["Y"]),
                                ],
                                "then",
                                [],
                                [_mkv_("Y")],
                            ),
                            else_branch=oh.make_graph(
                                [
                                    # not shadowing
                                    oh.make_node("Sub", ["X0", "three"], ["Y"])
                                ],
                                "else",
                                [],
                                [_mkv_("Y")],
                            ),
                        ),
                        oh.make_node("CastLike", ["Z_c", "X"], ["Z"]),
                    ],
                    "test",
                    [
                        oh.make_tensor_value_info("X", TensorProto.FLOAT, ["N"]),
                        oh.make_tensor_value_info("one", TensorProto.FLOAT, ["N"]),
                    ],
                    [oh.make_tensor_value_info("Z", TensorProto.UNDEFINED, ["N"])],
                    [
                        onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                        onh.from_array(np.array([2], dtype=np.float32), name="two"),
                    ],
                ),
                opset_imports=[oh.make_operatorsetid("", 18)],
                ir_version=10,
            )

        feeds = {
            "X": np.array([1, 2, 3], dtype=np.float32),
            "one": np.array([1], dtype=np.float32),
        }
        feeds2 = {
            "X": -np.array([1, 2, 3], dtype=np.float32),
            "one": np.array([1], dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "three.*SSA shadowing is not allowed"):
            GraphBuilder(_make_model())

        model = _make_model(shadowing=False)
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        expected = ref.run(None, feeds)[0]
        expected2 = ref.run(None, feeds2)[0]
        self.dump_onnx("test_remove_identity_shadowing.onnx", model)

        gr = GraphBuilder(model)
        self.assertEqual(len(gr.inputs), 2)
        self.assertEqual(len(gr.initializers_dict), 2)
        self.assertEqual(len(gr.outputs), 1)
        if_node = [n for n in gr.nodes if n.op_type == "If"][0]  # noqa: RUF015
        else_graph = next(att.g for att in if_node.attribute if att.name == "else_branch")
        self.assertEqual(else_graph.node[0].input, ["X0", "three"])
        gr.remove_identity_nodes()
        if_node = [n for n in gr.nodes if n.op_type == "If"][0]  # noqa: RUF015
        else_graph = next(att.g for att in if_node.attribute if att.name == "else_branch")
        self.assertEqual(else_graph.node[0].input, ["X0", "two"])
        self.assertEqual(len(gr.inputs), 2)
        self.assertEqual(len(gr.initializers_dict), 2)
        self.assertEqual(len(gr.outputs), 1)
        onx = gr.to_onnx()
        oc.check_model(onx.proto)
        self.dump_onnx("test_remove_identity_shadowing.opt.onnx", onx)

        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualAny(expected, got)
        got2 = ref2.run(None, feeds2)[0]
        self.assertEqualAny(expected2, got2)

        model = _make_model(shadowing=False)
        onx = self.call_optimizer(model)
        oc.check_model(onx.proto)
        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualAny(expected, got)
        got2 = ref2.run(None, feeds2)[0]
        self.assertEqualAny(expected2, got2)

    def test_remove_post_shadowing(self):
        def _mkv_(name):
            value_info_proto = onnx.ValueInfoProto()
            value_info_proto.name = name
            return value_info_proto

        def _make_model():
            return oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("ReduceSum", ["X"], ["Xred"]),
                        oh.make_node("Add", ["X", "two"], ["X0"]),
                        oh.make_node("Add", ["X0", "zero"], ["X00"]),
                        oh.make_node("CastLike", ["one", "Xred"], ["one_c"]),
                        oh.make_node("ReduceSum", ["X00"], ["Xred2"]),
                        oh.make_node("Add", ["Xred", "Xred2"], ["Xred3"]),
                        oh.make_node("Greater", ["Xred3", "one_c"], ["cond"]),
                        oh.make_node("Identity", ["two"], ["three"]),
                        oh.make_node(
                            "If",
                            ["cond"],
                            ["Z_c"],
                            then_branch=oh.make_graph(
                                [
                                    # shadowing
                                    oh.make_node("Constant", [], ["five"], value_floats=[2.1]),
                                    oh.make_node("Add", ["X00", "five"], ["Y"]),
                                ],
                                "then",
                                [],
                                [_mkv_("Y")],
                            ),
                            else_branch=oh.make_graph(
                                [
                                    # not shadowing
                                    oh.make_node("Identity", ["three"], ["four"]),
                                    oh.make_node("Sub", ["X0", "four"], ["Y"]),
                                ],
                                "else",
                                [],
                                [_mkv_("Y")],
                            ),
                        ),
                        oh.make_node("CastLike", ["Z_c", "X"], ["Zc"]),
                        oh.make_node("Neg", ["Zc"], ["five"]),
                        oh.make_node("Abs", ["five"], ["four"]),
                        oh.make_node(
                            "If",
                            ["cond"],
                            ["D"],
                            then_branch=oh.make_graph(
                                [
                                    # shadowing
                                    oh.make_node("Add", ["X00", "five"], ["Y"])
                                ],
                                "then",
                                [],
                                [_mkv_("Y")],
                            ),
                            else_branch=oh.make_graph(
                                [
                                    # not shadowing
                                    oh.make_node("Sub", ["X0", "three"], ["t"]),
                                    oh.make_node("Mul", ["t", "four"], ["Y"]),
                                ],
                                "else",
                                [],
                                [_mkv_("Y")],
                            ),
                        ),
                        oh.make_node("Add", ["five", "D"], ["Z"]),
                    ],
                    "test",
                    [
                        oh.make_tensor_value_info("X", TensorProto.FLOAT, ["N"]),
                        oh.make_tensor_value_info("one", TensorProto.FLOAT, ["N"]),
                    ],
                    [oh.make_tensor_value_info("Z", TensorProto.UNDEFINED, ["N"])],
                    [
                        onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                        onh.from_array(np.array([2], dtype=np.float32), name="two"),
                    ],
                ),
                opset_imports=[oh.make_operatorsetid("", 18)],
                ir_version=10,
            )

        feeds = {
            "X": np.array([1, 2, 3], dtype=np.float32),
            "one": np.array([1], dtype=np.float32),
        }
        feeds2 = {
            "X": -np.array([1, 2, 3], dtype=np.float32),
            "one": np.array([1], dtype=np.float32),
        }
        model = _make_model()
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        expected = ref.run(None, feeds)[0]
        expected2 = ref.run(None, feeds2)[0]

        gr = GraphBuilder(model)
        self.assertEqual(len(gr.inputs), 2)
        self.assertEqual(len(gr.initializers_dict), 2)
        self.assertEqual(len(gr.outputs), 1)
        gr.remove_identity_nodes()
        self.assertEqual(len(gr.inputs), 2)
        self.assertEqual(len(gr.initializers_dict), 2)
        self.assertEqual(len(gr.outputs), 1)
        onx = gr.to_onnx()
        oc.check_model(onx.proto)

        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualAny(expected, got)
        got2 = ref2.run(None, feeds2)[0]
        self.assertEqualAny(expected2, got2)

        model = _make_model()
        onx = self.call_optimizer(model)
        oc.check_model(onx.proto)
        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualAny(expected, got)
        got2 = ref2.run(None, feeds2)[0]
        self.assertEqualAny(expected2, got2)

    def test_remove_duplicated_shape_nodes(self):
        opset_imports = [oh.make_opsetid("", 18)]
        nodes = [
            oh.make_node("Shape", ["X"], ["s1"]),
            oh.make_node("Shape", ["X"], ["s2"]),
            oh.make_node("Add", ["s1", "s2"], ["z"]),
        ]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M"])],
            [oh.make_tensor_value_info("z", TensorProto.INT64, [2])],
        )
        model = oh.make_model(graph, opset_imports=opset_imports)
        self.assertEqual(["Shape", "Shape", "Add"], [n.op_type for n in model.graph.node])

        gr = GraphBuilder(model)
        n_removed = gr.inner_builder.remove_duplicate_nodes()
        self.assertEqual(1, n_removed)
        self.assertEqual(["Shape", "Add"], [n.op_type for n in gr.nodes])
        self.assertEqual(["s1", "s1"], list(gr.nodes[1].input))

        onx = gr.to_onnx()
        oc.check_model(onx.proto)

        ref = ExtendedReferenceEvaluator(onx)
        x = np.zeros((4, 5), dtype=np.float32)
        (z,) = ref.run(None, {"X": x})
        self.assertEqualArray(np.array([8, 10], dtype=np.int64), z)


if __name__ == "__main__":
    unittest.main(verbosity=2)
