import unittest
import ml_dtypes
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase, hide_stdout
from yobx.helpers._onnx_simple_text_plot import onnx_simple_text_plot
from yobx.helpers.onnx_helper import (
    attr_proto_to_python,
    get_hidden_inputs,
    make_model_with_local_functions,
    make_subfunction,
    enumerate_results,
    onnx_find,
    onnx_dtype_name,
    pretty_onnx,
    shadowing_names,
    tensor_dtype_to_np_dtype,
    same_function_proto,
)

TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64


class TestOnnxHelper(ExtTestCase):
    def _get_model(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        return model

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
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E4M3FN), ml_dtypes.float8_e4m3fn
        )
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E4M3FNUZ), ml_dtypes.float8_e4m3fnuz
        )
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E5M2), ml_dtypes.float8_e5m2
        )
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E5M2FNUZ), ml_dtypes.float8_e5m2fnuz
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
                [onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="reshape_shape")],
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
        att = oh.make_attribute("value", onh.from_array(np.array([1.0, 2.0], dtype=np.float32)))
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

    def test_get_hidden_inputs_no_hidden(self):
        # All inputs are declared; no hidden inputs expected
        graph = oh.make_graph(
            [oh.make_node("Add", ["X", "Y"], ["Z"])],
            "test",
            [
                oh.make_tensor_value_info("X", TFLOAT, [3, 4]),
                oh.make_tensor_value_info("Y", TFLOAT, [3, 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [3, 4])],
        )
        self.assertEqual(get_hidden_inputs(graph), set())

    def test_get_hidden_inputs_with_hidden(self):
        # A node references "outer_val" which is not declared as a graph input
        graph = oh.make_graph(
            [oh.make_node("Add", ["X", "outer_val"], ["Z"])],
            "test",
            [oh.make_tensor_value_info("X", TFLOAT, [3, 4])],
            [oh.make_tensor_value_info("Z", TFLOAT, [3, 4])],
        )
        self.assertEqual(get_hidden_inputs(graph), {"outer_val"})

    def test_get_hidden_inputs_empty_names_excluded(self):
        # Empty string inputs (optional/absent inputs) must not appear in the result
        graph = oh.make_graph(
            [oh.make_node("Add", ["X", ""], ["Z"])],
            "test",
            [oh.make_tensor_value_info("X", TFLOAT, [3, 4])],
            [oh.make_tensor_value_info("Z", TFLOAT, [3, 4])],
        )
        self.assertEqual(get_hidden_inputs(graph), set())

    def test_get_hidden_inputs_subgraph(self):
        # A subgraph (e.g., an If branch) may reference variables from the outer graph.
        # get_hidden_inputs on the inner graph returns those outer variables.
        # get_hidden_inputs on the outer graph filters them out when they are in scope.
        inner_graph = oh.make_graph(
            [oh.make_node("Identity", ["outer_val"], ["res"])],
            "inner",
            [],
            [oh.make_tensor_value_info("res", TFLOAT, [3, 4])],
        )
        if_node = oh.make_node(
            "If", ["cond"], ["result"], then_branch=inner_graph, else_branch=inner_graph
        )
        outer_graph = oh.make_graph(
            [if_node],
            "outer",
            [
                oh.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
                oh.make_tensor_value_info("outer_val", TFLOAT, [3, 4]),
            ],
            [oh.make_tensor_value_info("result", TFLOAT, [3, 4])],
        )
        # "outer_val" is declared in the outer graph, so it is not hidden there
        self.assertEqual(get_hidden_inputs(outer_graph), set())
        # "outer_val" is NOT declared in the inner graph, so it is hidden there
        self.assertEqual(get_hidden_inputs(inner_graph), {"outer_val"})

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
                [onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="reshape_shape")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        text = onnx_simple_text_plot(model, add_links=True)
        self.assertIn("Reshape(concat_out, reshape_shape) -> Z", text)
        # add_links=True renders ASCII art links; intermediate link lines end with '|'
        self.assertTrue(any(line.endswith("|") for line in text.splitlines()))

    def test_onnx_find(self):
        model = self._get_model()
        res = onnx_find(model, watch={"xm2"})
        self.assertEqual(len(res), 2)
        self.assertIn("xm2", res[0].output)
        self.assertIn("xm2", res[1].input)

    @hide_stdout()
    def test_enumerate_results(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        res = list(enumerate_results(model, "xu1", verbose=2))
        ress = ";".join(str(r) for r in res)
        self.assertEqual(
            ">> xu1 - (0:Unsqueeze:) :: Unsqueeze(X, zero) -> xu1;"
            "<< xu1 - (1:Unsqueeze:) :: Unsqueeze(xu1, un) -> xu2",
            ress,
        )
        self.assertEqual(2, len(list(enumerate_results(model, "shape1", verbose=2))))
        self.assertEqual(2, len(list(enumerate_results(model, "X", verbose=2))))
        self.assertEqual(2, len(list(enumerate_results(model, "Z", verbose=2))))

    @hide_stdout()
    def test_enumerate_results_loop(self):
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

        model = oh.make_model(
            graph=oh.make_graph(
                name="loop_test",
                inputs=[
                    oh.make_tensor_value_info("trip_count", TINT64, ["a"]),
                    oh.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [1]),
                ],
                outputs=[oh.make_tensor_value_info("res", TFLOAT, [])],
                nodes=[
                    oh.make_node("SequenceEmpty", [], ["seq_empty"], dtype=TFLOAT),
                    oh.make_node(
                        "Loop",
                        inputs=["trip_count", "cond", "seq_empty"],
                        outputs=["seq_res"],
                        body=oh.make_graph(
                            [
                                oh.make_node(
                                    "Identity", inputs=["cond_in"], outputs=["cond_out"]
                                ),
                                oh.make_node(
                                    "Constant",
                                    inputs=[],
                                    outputs=["x"],
                                    value=oh.make_tensor(
                                        name="const_tensor_x",
                                        data_type=TFLOAT,
                                        dims=x.shape,
                                        vals=x.flatten().astype(float),
                                    ),
                                ),
                                oh.make_node(
                                    "Constant",
                                    inputs=[],
                                    outputs=["one"],
                                    value=oh.make_tensor(
                                        name="const_tensor_one",
                                        data_type=TINT64,
                                        dims=(),
                                        vals=[1],
                                    ),
                                ),
                                oh.make_node(
                                    "Constant",
                                    inputs=[],
                                    outputs=["slice_start"],
                                    value=oh.make_tensor(
                                        name="const_tensor_zero",
                                        data_type=TINT64,
                                        dims=(1,),
                                        vals=[0],
                                    ),
                                ),
                                oh.make_node(
                                    "Add", inputs=["iter_count", "one"], outputs=["end"]
                                ),
                                oh.make_node(
                                    "Constant",
                                    inputs=[],
                                    outputs=["axes"],
                                    value=oh.make_tensor(
                                        name="const_tensor_axes",
                                        data_type=TINT64,
                                        dims=(1,),
                                        vals=[0],
                                    ),
                                ),
                                oh.make_node(
                                    "Unsqueeze", inputs=["end", "axes"], outputs=["slice_end"]
                                ),
                                oh.make_node(
                                    "Slice",
                                    inputs=["x", "slice_start", "slice_end"],
                                    outputs=["slice_out"],
                                ),
                                oh.make_node(
                                    "SequenceInsert",
                                    inputs=["seq_in", "slice_out"],
                                    outputs=["seq_out"],
                                ),
                            ],
                            "loop_body",
                            [
                                oh.make_tensor_value_info("iter_count", TINT64, []),
                                oh.make_tensor_value_info("cond_in", onnx.TensorProto.BOOL, []),
                                oh.make_tensor_sequence_value_info("seq_in", TFLOAT, None),
                            ],
                            [
                                oh.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, []),
                                oh.make_tensor_sequence_value_info("seq_out", TFLOAT, None),
                            ],
                        ),
                    ),
                    oh.make_node(
                        "ConcatFromSequence",
                        inputs=["seq_res"],
                        outputs=["res"],
                        axis=0,
                        new_axis=0,
                    ),
                ],
            ),
            ir_version=10,
            opset_imports=[oh.make_opsetid("", 22)],
        )
        res = list(enumerate_results(model, "slice_start", verbose=2))
        self.assertEqual(len(res), 2)

    def test_shadowing_names(self):
        def _mkv_(name):
            value_info_proto = onnx.ValueInfoProto()
            value_info_proto.name = name
            return value_info_proto

        model = oh.make_model(
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
                                # shadowing
                                oh.make_node("Constant", [], ["three"], value_floats=[2.1]),
                                oh.make_node("Add", ["X00", "three"], ["Y"]),
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
                    oh.make_tensor_value_info("X", TFLOAT, ["N"]),
                    oh.make_tensor_value_info("one", TFLOAT, ["N"]),
                ],
                [oh.make_tensor_value_info("Z", onnx.TensorProto.UNDEFINED, ["N"])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([2], dtype=np.float32), name="two"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 18)],
            ir_version=10,
        )
        self.assertEqual(
            ({"three"}, set(), {"cond", "Z", "X0", "Z_c", "three", "one_c", "Xred", "X00", "Y"}),
            shadowing_names(model),
        )

    def test_make_model_with_local_functions(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        for i_node in [0, 1, 2, 3]:
            node = model.graph.node[i_node]
            meta = node.metadata_props.add()
            meta.key = "namespace"
            meta.value = "LLL"
        new_model = make_model_with_local_functions(model, "^LLL$")
        onnx.checker.check_model(model)
        self.assertEqual(len(new_model.functions), 1)
        self.assertEqual(
            ["X", "Y", "shape1", "shape2", "un", "zero"], new_model.functions[0].input
        )
        self.assertEqual(["xm1", "xm2c"], new_model.functions[0].output)
        self.assertEqual("LLL", new_model.functions[0].name)
        self.assertEqual("local_function", new_model.functions[0].domain)
        self.assertIn("LLL[local_function]", pretty_onnx(new_model))
        onnx.checker.check_model(new_model)

    def test_make_model_with_local_functions_bug(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        for i_node in [0, 2, 3, 4]:
            node = model.graph.node[i_node]
            meta = node.metadata_props.add()
            meta.key = "namespace"
            meta.value = "LLL"
        self.assertRaise(
            lambda: make_model_with_local_functions(model, "^LLL$", allow_extensions=False),
            ValueError,
        )
        onnx.checker.check_model(model)

    @hide_stdout()
    def test_make_model_with_local_functions_2(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        for i_node in [0, 1, 2, 3]:
            node = model.graph.node[i_node]
            meta = node.metadata_props.add()
            meta.key = f"source[{i_node}]"
            meta.value = f"LLL{i_node//3}"
        new_model = make_model_with_local_functions(
            model, "^LLL[01]$", metadata_key_prefix="source[", verbose=1
        )
        onnx.checker.check_model(model)
        self.assertEqual(len(new_model.functions), 2)
        p = pretty_onnx(new_model)
        self.assertIn("LLL0[local_function]", p)
        self.assertIn("LLL1[local_function]", p)

        self.assertEqual(["X", "shape1", "un", "zero"], new_model.functions[0].input)
        self.assertEqual(["xm1"], new_model.functions[0].output)
        self.assertEqual("LLL0", new_model.functions[0].name)
        self.assertEqual("local_function", new_model.functions[0].domain)
        self.assertEqual(len(new_model.functions[0].node), 3)

        self.assertEqual(["Y", "shape2"], new_model.functions[1].input)
        self.assertEqual(["xm2c"], new_model.functions[1].output)
        self.assertEqual("LLL1", new_model.functions[1].name)
        self.assertEqual("local_function", new_model.functions[1].domain)
        self.assertEqual(len(new_model.functions[1].node), 1)

        onnx.checker.check_model(new_model)

    @hide_stdout()
    def test_make_model_with_local_functions_3(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        for i_node in range(len(model.graph.node) - 1):
            if i_node == 2:
                continue
            node = model.graph.node[i_node]
            meta = node.metadata_props.add()
            meta.key = f"source[{i_node}]"
            meta.value = "LLL"
        self.assertRaise(
            lambda: make_model_with_local_functions(
                model, "^LLL$", metadata_key_prefix="source[", verbose=1, allow_extensions=False
            ),
            ValueError,
        )
        new_model = make_model_with_local_functions(
            model, "^LLL$", metadata_key_prefix="source[", verbose=1
        )
        onnx.checker.check_model(new_model)
        self.assertEqual(len(new_model.functions), 1)
        p = pretty_onnx(new_model)
        self.assertIn("LLL[local_function]", p)

        self.assertEqual(
            ["X", "Y", "shape1", "shape2", "un", "zero"], new_model.functions[0].input
        )
        self.assertEqual(["xm"], new_model.functions[0].output)
        self.assertEqual("LLL", new_model.functions[0].name)
        self.assertEqual("local_function", new_model.functions[0].domain)
        self.assertEqual(len(new_model.functions[0].node), 6)
        onnx.checker.check_model(new_model)

    def test_make_subfunction(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        new_function = make_subfunction(
            "localf",
            model.graph.node[:4],
            opset_imports=model.opset_import,
            output_names=["xm1", "xm2c"],
        )
        self.assertIsInstance(new_function, onnx.FunctionProto)
        self.assertEqual(len(new_function.node), 4)
        self.assertEqual(new_function.output, ["xm1", "xm2c"])
        self.assertEqual(new_function.input, ["X", "Y", "shape1", "shape2", "un", "zero"])

    def test_same_function_proto(self):
        f1 = oh.make_function(
            "custom",
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [oh.make_node("MatMul", ["x", "a"], ["xa"]), oh.make_node("Add", ["xa", "b"], ["y"])],
            [oh.make_opsetid("", 14)],
            [],
        )
        self.assertEqualTrue(same_function_proto(f1, f1, verbose=1))
        f2 = oh.make_function(
            "custom",
            "LinearRegression",
            ["x_", "a_", "b_"],
            ["y_"],
            [
                oh.make_node("MatMul", ["x_", "a_"], ["xa_"]),
                oh.make_node("Add", ["xa_", "b_"], ["y_"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )
        self.assertEqualTrue(same_function_proto(f1, f2, verbose=1))
        f3 = oh.make_function(
            "custom",
            "LinearRegression",
            ["x_", "a_", "b_"],
            ["y_"],
            [
                oh.make_node("MatMul", ["x_", "a_"], ["xa_"]),
                oh.make_node("Add", ["xb_", "a_"], ["y_"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )
        self.assertEqual(
            same_function_proto(f1, f3, verbose=1),
            "different input names at node 1, ['xa', 'b'], ['xb_', 'a_'] != ['xa_', 'b_']",
        )


class TestAttrProtoToPython(ExtTestCase):
    def test_float(self):
        attr = oh.make_attribute("alpha", 0.5)
        self.assertAlmostEqual(attr_proto_to_python(attr), 0.5)

    def test_int(self):
        attr = oh.make_attribute("axis", 1)
        self.assertEqual(attr_proto_to_python(attr), 1)

    def test_string(self):
        attr = oh.make_attribute("mode", "linear")
        self.assertEqual(attr_proto_to_python(attr), "linear")

    def test_string_bytes(self):
        attr = onnx.AttributeProto()
        attr.name = "mode"
        attr.type = onnx.AttributeProto.STRING
        attr.s = b"linear"
        self.assertEqual(attr_proto_to_python(attr), "linear")

    def test_tensor(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        attr = oh.make_attribute("value", onh.from_array(arr))
        result = attr_proto_to_python(attr)
        np.testing.assert_array_equal(result, arr)

    def test_floats(self):
        attr = oh.make_attribute("scales", [1.0, 2.0, 3.0])
        self.assertEqual(attr_proto_to_python(attr), [1.0, 2.0, 3.0])

    def test_ints(self):
        attr = oh.make_attribute("pads", [0, 1, 0, 1])
        self.assertEqual(attr_proto_to_python(attr), [0, 1, 0, 1])

    def test_strings(self):
        attr = oh.make_attribute("keys", ["a", "b", "c"])
        self.assertEqual(attr_proto_to_python(attr), ["a", "b", "c"])

    def test_unsupported_raises(self):
        attr = onnx.AttributeProto()
        attr.name = "body"
        attr.type = onnx.AttributeProto.GRAPH
        self.assertRaises(NotImplementedError, attr_proto_to_python, attr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
