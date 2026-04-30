import itertools
import os
import unittest
from typing import Optional
import numpy as np
from onnx import (
    ModelProto,
    TensorProto,
    helper as oh,
    numpy_helper as onh,
    load as onnx_load,
    shape_inference,
)
from onnx.checker import check_model
from yobx.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    requires_cuda,
    requires_onnxruntime,
    requires_torch,
    hide_stdout,
)
from yobx.xbuilder.graph_builder import GraphBuilder, OptimizationOptions, InferShapesOptions
from yobx.xoptim import get_pattern_list
from yobx.xoptim.patterns_ort.activation import GeluErfPattern
from yobx.helpers.onnx_helper import choose_consistent_domain_opset, compatible_opsets
from yobx.reference import ExtendedReferenceEvaluator

TFLOAT = TensorProto.FLOAT
TINT64 = TensorProto.INT64


class TestGraphPatternOptimizationOrt(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_get_pattern_list(self):
        res = get_pattern_list("onnxruntime")
        names = set(r.__class__.__name__ for r in res)
        self.assertNotIn("ConstantScatterNDPattern", names)

    def test_choose_consistent_domain_opset(self):
        self.assertIsInstance(choose_consistent_domain_opset(""), int)
        self.assertEqual(choose_consistent_domain_opset("", {"": 10}), 10)
        self.assertEqual(choose_consistent_domain_opset("ai.onnx.ml", {"": 18}), 3)
        self.assertEqual(choose_consistent_domain_opset("com.microsoft", {"": 18}), 1)
        self.assertIsInstance(choose_consistent_domain_opset("", {"com.microsoft": 1}), int)
        self.assertRaise(
            lambda: choose_consistent_domain_opset("", {"ai.onnx.ml": 10}), AssertionError
        )

    @skipif_ci_windows("get_all_schemas_with_history returns wrong values")
    def test_compatible_opsets(self):
        self.assertTrue(compatible_opsets("", "Slice", 18, 18))
        self.assertTrue(compatible_opsets("", "Slice", 18, 17))
        self.assertFalse(compatible_opsets("", "Slice", 12, 13))
        self.assertFalse(compatible_opsets("", "Slice", 13, 12))
        self.assertFalse(compatible_opsets("", "Slice", 11, 13))
        self.assertTrue(compatible_opsets("", "Slice", 11, 12))
        self.assertFalse(compatible_opsets("", "Slice", 18, 1))

    def _get_model(self, name: str) -> ModelProto:
        p = os.path.join(os.path.dirname(__file__), "..", "xbuilder", "data", name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(__file__), "data", name)
        self.assertExists(p)
        return onnx_load(p)

    def common_fused_matmul(self, side):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xm1"], perm=[0, 1, 3, 2]),
                    oh.make_node(
                        "MatMul", ["xm1", "Y"] if side == "left" else ["Y", "xm1"], ["Z"]
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info(
                        "X", TFLOAT, [2, 2, 128, 32] if side == "left" else [2, 2, 32, 128]
                    ),
                    oh.make_tensor_value_info(
                        "Y", TFLOAT, [2, 2, 128, 64] if side == "left" else [2, 2, 64, 128]
                    ),
                ],
                [
                    oh.make_tensor_value_info(
                        "Z", TFLOAT, [2, 2, 32, 64] if side == "left" else [2, 2, 64, 32]
                    )
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = (
            {"X": self._range(2, 2, 128, 32), "Y": self._range(2, 2, 128, 64)}
            if side == "left"
            else {"X": self._range(2, 2, 32, 128), "Y": self._range(2, 2, 64, 128)}
        )
        if side == "left":
            assert feeds["X"][0, 0].T @ feeds["Y"][0, 0] is not None
        else:
            assert feeds["Y"][0, 0] @ feeds["X"][0, 0].T is not None
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FusedMatMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["FusedMatMul"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_fused_matmul_left(self):
        self.common_fused_matmul("left")

    def test_fused_matmul_right(self):
        self.common_fused_matmul("right")

    def test_fused_matmul_both(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xm1"], perm=[0, 1, 3, 2]),
                    oh.make_node("Transpose", ["Y"], ["ym1"], perm=[0, 1, 3, 2]),
                    oh.make_node("MatMul", ["xm1", "ym1"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 128, 32]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 64, 128]),
                ],
                [
                    oh.make_tensor_value_info("Z", TFLOAT, [2, 2, 32, 64]),
                    oh.make_tensor_value_info("xm1", TFLOAT, [2, 2, 32, 128]),
                    oh.make_tensor_value_info("ym1", TFLOAT, [2, 2, 128, 64]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 128, 32), "Y": self._range(2, 2, 64, 128)}
        assert feeds["X"][0, 0].T @ feeds["Y"][0, 0].T is not None
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FusedMatMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Transpose", "Transpose", "FusedMatMul"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])
        node = opt_onx.graph.node[2]
        self.assertEqual(node.op_type, "FusedMatMul")
        self.assertEqual(node.domain, "com.microsoft")
        for att in node.attribute:
            if att.name == "transA":
                self.assertEqual(att.i, 0)
            elif att.name == "transB":
                self.assertEqual(att.i, 1)

    def test_fused_matmul_both_div(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["xm1"], perm=[0, 1, 3, 2]),
                    oh.make_node("Transpose", ["Y"], ["ym1"], perm=[0, 1, 3, 2]),
                    oh.make_node("MatMul", ["xm1", "ym1"], ["zd"]),
                    oh.make_node("Div", ["zd", "deux"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 128, 32]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 64, 128]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 2, 32, 64])],
                [onh.from_array(np.array([2], dtype=np.float32), name="deux")],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 128, 32), "Y": self._range(2, 2, 64, 128)}
        assert feeds["X"][0, 0].T @ feeds["Y"][0, 0].T is not None
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FusedMatMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Transpose", "FusedMatMul"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])
        node = opt_onx.graph.node[1]
        self.assertEqual(node.op_type, "FusedMatMul")
        self.assertEqual(node.domain, "com.microsoft")
        for att in node.attribute:
            if att.name == "transA":
                self.assertEqual(att.i, 0)
            elif att.name == "transB":
                self.assertEqual(att.i, 1)
            elif att.name == "alpha":
                self.assertEqual(att.f, 0.5)

    def test_fused_matmul_div(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "FusedMatMul", ["X", "Y"], ["zd"], domain="com.microsoft", alpha=1.3
                    ),
                    oh.make_node("Div", ["zd", "deux"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 32, 128]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 128, 64]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 2, 32, 64])],
                [onh.from_array(np.array([2], dtype=np.float32), name="deux")],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 32, 128), "Y": self._range(2, 2, 128, 64)}
        assert feeds["X"][0, 0] @ feeds["Y"][0, 0] is not None
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FusedMatMulDiv"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["FusedMatMul"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "FusedMatMul")
        self.assertEqual(node.domain, "com.microsoft")
        for att in node.attribute:
            if att.name == "transA":
                self.assertEqual(att.i, 0)
            elif att.name == "transB":
                self.assertEqual(att.i, 0)
            elif att.name == "alpha":
                self.assertAlmostEqual(att.f, 1.3 / 2, atol=1e-5)

    def get_simplified_layer_normalization_model(self, div, dyn):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Pow", ["X", "exp"], ["x2"]),
                    oh.make_node("ReduceMean", ["x2", "axis"], ["xr"]),
                    oh.make_node("Add", ["xr", "eps"], ["xa"]),
                    oh.make_node("Sqrt", ["xa"], ["xq"]),
                    (
                        oh.make_node("Div", ["one", "xq"], ["xi"])
                        if div
                        else oh.make_node("Reciprocal", ["xq"], ["xi"])
                    ),
                    oh.make_node("Mul", ["xi", "X"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "D" if dyn else 4])],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "D" if dyn else 4])],
                [
                    onh.from_array(np.array([2], dtype=np.float32), name="exp"),
                    onh.from_array(
                        np.array([9.999999974752427e-7], dtype=np.float32), name="eps"
                    ),
                    onh.from_array(np.array([-1], dtype=np.int64), name="axis"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_simplified_layer_normalization_model(self):
        for div, dyn in itertools.product([False, True], [False, True]):
            with self.subTest(div=div, dyn=dyn):
                model = self.get_simplified_layer_normalization_model(div=div, dyn=dyn)
                gr = GraphBuilder(
                    model,
                    infer_shapes_options=True,
                    optimization_options=OptimizationOptions(
                        patterns=["SimplifiedLayerNormalization"]
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(
                    (
                        ["Shape", "Gather", "ConstantOfShape", "SimplifiedLayerNormalization"]
                        if dyn
                        else ["SimplifiedLayerNormalization"]
                    ),
                    [n.op_type for n in opt_onx.graph.node],
                )

                feeds = {"X": np.arange(20).reshape((5, 4)).astype(np.float32)}
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)

                ninits = {(False, False): 1, (False, True): 1, (True, False): 1, (True, True): 1}
                self.assertEqual(ninits[div, dyn], len(opt_onx.graph.initializer))

                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                self.assertEqualArray(expected[0], got[0], atol=1e-5)

                if got:
                    from onnxruntime import InferenceSession

                    sess = InferenceSession(
                        opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
                    )
                    got = sess.run(None, feeds)
                    self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def get_simplified_layer_normalization_model_eps_first(self, dyn):
        """Build a SimplifiedLayerNorm model where epsilon is at Add.input[0]."""
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Pow", ["X", "exp"], ["x2"]),
                    oh.make_node("ReduceMean", ["x2", "axis"], ["xr"]),
                    # epsilon at input[0], ReduceMean output at input[1]
                    oh.make_node("Add", ["eps", "xr"], ["xa"]),
                    oh.make_node("Sqrt", ["xa"], ["xq"]),
                    oh.make_node("Reciprocal", ["xq"], ["xi"]),
                    oh.make_node("Mul", ["xi", "X"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "D" if dyn else 4])],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "D" if dyn else 4])],
                [
                    onh.from_array(np.array([2], dtype=np.float32), name="exp"),
                    onh.from_array(
                        np.array([9.999999974752427e-7], dtype=np.float32), name="eps"
                    ),
                    onh.from_array(np.array([-1], dtype=np.int64), name="axis"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_simplified_layer_normalization_eps_first(self):
        """Test SimplifiedLayerNormalizationPattern when epsilon is at Add.input[0]."""
        for dyn in [False, True]:
            with self.subTest(dyn=dyn):
                model = self.get_simplified_layer_normalization_model_eps_first(dyn=dyn)
                gr = GraphBuilder(
                    model,
                    infer_shapes_options=True,
                    optimization_options=OptimizationOptions(
                        patterns=["SimplifiedLayerNormalization"]
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(
                    (
                        ["Shape", "Gather", "ConstantOfShape", "SimplifiedLayerNormalization"]
                        if dyn
                        else ["SimplifiedLayerNormalization"]
                    ),
                    [n.op_type for n in opt_onx.graph.node],
                )

                feeds = {"X": np.arange(20).reshape((5, 4)).astype(np.float32)}
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)

                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def get_simplified_layer_normalization_model_output(self, div, dyn):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Pow", ["X", "exp"], ["x2"]),
                    oh.make_node("ReduceMean", ["x2", "axis"], ["xr"]),
                    oh.make_node("Add", ["xr", "eps"], ["xa"]),
                    oh.make_node("Sqrt", ["xa"], ["xq"]),
                    (
                        oh.make_node("Div", ["one", "xq"], ["Z"])
                        if div
                        else oh.make_node("Reciprocal", ["xq"], ["Z"])
                    ),
                    oh.make_node("Mul", ["Z", "X"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "D" if dyn else 4])],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "D" if dyn else 4]),
                    oh.make_tensor_value_info("Z", TFLOAT, ["a", 1]),
                ],
                [
                    onh.from_array(np.array([2], dtype=np.float32), name="exp"),
                    onh.from_array(
                        np.array([9.999999974752427e-7], dtype=np.float32), name="eps"
                    ),
                    onh.from_array(np.array([-1], dtype=np.int64), name="axis"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_simplified_layer_normalization_model_output(self):
        for div, dyn in itertools.product([False, True], [False, True]):
            with self.subTest(div=div, dyn=dyn):
                model = self.get_simplified_layer_normalization_model_output(div=div, dyn=dyn)
                gr = GraphBuilder(
                    model,
                    infer_shapes_options=True,
                    optimization_options=OptimizationOptions(
                        patterns=["SimplifiedLayerNormalization"]
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(
                    (
                        ["Shape", "Gather", "ConstantOfShape", "SimplifiedLayerNormalization"]
                        if dyn
                        else ["SimplifiedLayerNormalization"]
                    ),
                    [n.op_type for n in opt_onx.graph.node],
                )

                feeds = {"X": np.arange(20).reshape((5, 4)).astype(np.float32)}
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)

                ninits = {(False, False): 1, (False, True): 1, (True, False): 1, (True, True): 1}
                self.assertEqual(ninits[div, dyn], len(opt_onx.graph.initializer))

                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                self.assertEqualArray(expected[0], got[0], atol=1e-5)
                self.assertEqualArray(expected[1], got[1], atol=1e-5)

                if got:
                    from onnxruntime import InferenceSession

                    sess = InferenceSession(
                        opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
                    )
                    got = sess.run(None, feeds)
                    self.assertEqualArray(expected[0], got[0], atol=1e-5)
                    self.assertEqualArray(expected[1], got[1], atol=1e-5)

    def test_fused_matmul_both_div_2x(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Div", ["X", "deux"], ["half"]),
                    oh.make_node(
                        "FusedMatMul",
                        ["half", "X"],
                        ["x1"],
                        transA=1,
                        alpha=50.1,
                        domain="com.microsoft",
                    ),
                    oh.make_node(
                        "FusedMatMul",
                        ["X", "half"],
                        ["x2"],
                        transA=1,
                        alpha=0.07,
                        domain="com.microsoft",
                    ),
                    oh.make_node("Add", ["x1", "x2"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 4, 4]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 4, 4]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 2, 32, 64])],
                [onh.from_array(np.array([2], dtype=np.float32), name="deux")],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        feeds = {"X": self._range(2, 2, 4, 4), "Y": self._range(2, 2, 4, 4)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FusedMatMulx2"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["FusedMatMul", "FusedMatMul", "Add"], [n.op_type for n in opt_onx.graph.node]
        )
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_fused_matmul_transpose(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "FusedMatMul",
                        ["X", "Y"],
                        ["xy"],
                        transA=1,
                        transB=1,
                        alpha=50.1,
                        domain="com.microsoft",
                    ),
                    oh.make_node("Transpose", ["xy"], ["Z"], perm=[0, 1, 3, 2]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 6, 3]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 5, 6]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 2, None, None])],
                [onh.from_array(np.array([2], dtype=np.float32), name="deux")],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        feeds = {"X": self._range(2, 2, 6, 3), "Y": self._range(2, 2, 5, 6)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FusedMatMulTranspose"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["FusedMatMul"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    @ignore_warnings(UserWarning)
    def test_fast_gelu(self):
        data = os.path.join(os.path.dirname(__file__), "data", "layernorm.onnx")
        model = onnx_load(data, load_external_data=False)
        del model.opset_import[:]
        model.opset_import.append(oh.make_opsetid("", 20))
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["Cast", "Gelu", "FastGelu"], verbose=0, constant_folding=False
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("Gelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertIn("FastGelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertEqual(42, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

    def test_fast_gelu18(self):
        data = os.path.join(os.path.dirname(__file__), "data", "layernorm.onnx")
        model = onnx_load(data, load_external_data=False)
        del model.opset_import[:]
        model.opset_import.append(oh.make_opsetid("", 18))
        inputs = [tuple(n.input) for n in model.graph.node]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["Cast", "GeluOrt", "FastGelu"], verbose=0, constant_folding=False
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("Gelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertIn("FastGelu", set(n.op_type for n in opt_onx.graph.node))
        self.assertEqual(42, len(opt_onx.graph.initializer))
        new_inputs = [tuple(n.input) for n in opt_onx.graph.node]
        self.assertNotEqual(inputs, new_inputs)

    def test_bias_gelu(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "B"], ["xb"]),
                    oh.make_node("Div", ["xb", "sq2"], ["xbinv"]),
                    oh.make_node("Erf", ["xbinv"], ["xerf"]),
                    oh.make_node("Add", ["xerf", "one"], ["xerf1"]),
                    oh.make_node("Mul", ["xb", "xerf1"], ["y2"]),
                    oh.make_node("Mul", ["y2", "half"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 2, 4, 8])],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 4, 8])],
                [
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                    onh.from_array(np.array([0.5], dtype=np.float32), name="half"),
                    onh.from_array(np.array([1.4140625], dtype=np.float32), name="sq2"),
                    onh.from_array(
                        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.4, -0.1], dtype=np.float32),
                        name="B",
                    ),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 4, 8)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["BiasGelu"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["BiasGelu"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "BiasGelu")

    def test_bias_gelu_with_conflict(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "B"], ["xb"]),
                    oh.make_node("Div", ["xb", "sq2"], ["xbinv"]),
                    oh.make_node("Erf", ["xbinv"], ["xerf"]),
                    oh.make_node("Add", ["xerf", "one"], ["xerf1"]),
                    oh.make_node("Mul", ["xb", "xerf1"], ["y2"]),
                    oh.make_node("Mul", ["y2", "half"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 2, 4, 8])],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 4, 8])],
                [
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                    onh.from_array(np.array([0.5], dtype=np.float32), name="half"),
                    onh.from_array(np.array([1.4140625], dtype=np.float32), name="sq2"),
                    onh.from_array(
                        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.4, -0.1], dtype=np.float32),
                        name="B",
                    ),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 4, 8)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["AddAddMulMul", "AddAddMulMulBroadcast", "BiasGelu"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["BiasGelu"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "BiasGelu")

    def test_bias_split_gelu(self):
        # Shape: X is (2, 4, 8), bias is (8).
        # After Add, Split along last dim produces two halves of size 4.
        # t1 (left) is multiplied by Gelu(t2) (right).
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "B"], ["xb"]),
                    oh.make_node("Split", ["xb"], ["xb1", "xb2"], axis=-1, num_outputs=2),
                    oh.make_node("Div", ["xb2", "sq2"], ["xb2d"]),
                    oh.make_node("Erf", ["xb2d"], ["xb2e"]),
                    oh.make_node("Add", ["xb2e", "one"], ["xb2e1"]),
                    oh.make_node("Mul", ["xb2", "xb2e1"], ["xb2g"]),
                    oh.make_node("Mul", ["xb2g", "half"], ["xb2gelu"]),
                    oh.make_node("Mul", ["xb1", "xb2gelu"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 4, 8])],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 4, 4])],
                [
                    onh.from_array(
                        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.4, -0.1], dtype=np.float32),
                        name="B",
                    ),
                    onh.from_array(np.array([1.4140625], dtype=np.float32), name="sq2"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                    onh.from_array(np.array([0.5], dtype=np.float32), name="half"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["BiasSplitGelu"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["BiasSplitGelu"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "BiasSplitGelu")
        self.assertEqual(node.domain, "com.microsoft")

    @requires_cuda()
    def test_bias_split_gelu_cuda(self):
        from onnxruntime import InferenceSession

        # Shape: X is (2, 4, 8), bias is (8).
        # After Add, Split along last dim produces two halves of size 4.
        # t1 (left) is multiplied by Gelu(t2) (right).
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "B"], ["xb"]),
                    oh.make_node("Split", ["xb"], ["xb1", "xb2"], axis=-1, num_outputs=2),
                    oh.make_node("Div", ["xb2", "sq2"], ["xb2d"]),
                    oh.make_node("Erf", ["xb2d"], ["xb2e"]),
                    oh.make_node("Add", ["xb2e", "one"], ["xb2e1"]),
                    oh.make_node("Mul", ["xb2", "xb2e1"], ["xb2g"]),
                    oh.make_node("Mul", ["xb2g", "half"], ["xb2gelu"]),
                    oh.make_node("Mul", ["xb1", "xb2gelu"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 4, 8])],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 4, 4])],
                [
                    onh.from_array(
                        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.4, -0.1], dtype=np.float32),
                        name="B",
                    ),
                    onh.from_array(np.array([1.4140625], dtype=np.float32), name="sq2"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                    onh.from_array(np.array([0.5], dtype=np.float32), name="half"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 4, 8)}
        ref = InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["BiasSplitGelu"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["BiasSplitGelu"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "BiasSplitGelu")

    def test_gelu_erf(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Div", ["X", "sq2"], ["xd"]),
                    oh.make_node("Erf", ["xd"], ["exd"]),
                    oh.make_node("Add", ["exd", "one"], ["aexd"]),
                    oh.make_node("Mul", ["X", "aexd"], ["y2"]),
                    oh.make_node("Mul", ["half", "y2"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 2, 4, 8])],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 4, 8])],
                [
                    onh.from_array(np.array([1.4140625], dtype=np.float32), name="sq2"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                    onh.from_array(np.array([0.5], dtype=np.float32), name="half"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(2, 2, 4, 8)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=[GeluErfPattern(verbose=0)], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["Gelu"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "Gelu")

    @requires_cuda()
    def test_bias_softmax(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Softmax", ["xy"], ["Z"], axis=-1),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [16, 8, 4, 8]),
                    oh.make_tensor_value_info("Y", TFLOAT, [16, 1, 4, 8]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [16, 8, 4, 8])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"X": self._range(16, 8, 4, 8), "Y": self._range(16, 1, 4, 8)}
        ref = InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["BiasSoftmax"], processor="CPU,CUDA", verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["BiasSoftmax"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)

    def test_fused_conv(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Conv",
                        ["X", "W", "B"],
                        ["c"],
                        dilations=[1, 1],
                        group=1,
                        pads=[1, 1, 1, 1],
                        strides=[1, 1],
                    ),
                    oh.make_node("Relu", ["c"], ["Y"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, 8, 6, 6]),
                    oh.make_tensor_value_info("W", TFLOAT, [8, 8, 3, 3]),
                    oh.make_tensor_value_info("B", TFLOAT, [8]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, [1, 8, 6, 6])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        feeds = {"X": self._range(1, 8, 6, 6), "W": self._range(8, 8, 3, 3), "B": self._range(8)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["FusedConv"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["FusedConv"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_quick_gelu(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Sigmoid", ["X"], ["S"]), oh.make_node("Mul", ["X", "S"], ["Y"])],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [1, 8, 6, 6])],
                [oh.make_tensor_value_info("Y", TFLOAT, [1, 8, 6, 6])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        feeds = {"X": self._range(1, 8, 6, 6)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["QuickGelu"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["QuickGelu"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_gemm_fast_gelu_with_bias(self):
        from yobx.reference import ExtendedReferenceEvaluator

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["A", "B"], ["ab"]),
                    oh.make_node("Add", ["ab", "bias"], ["ab_bias"]),
                    oh.make_node("FastGelu", ["ab_bias"], ["Y"], domain="com.microsoft"),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("A", TFLOAT, [2, 4]),
                    oh.make_tensor_value_info("B", TFLOAT, [4, 8]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 8])],
                [
                    onh.from_array(
                        np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.4, 0.0, -0.3], dtype=np.float32),
                        name="bias",
                    )
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"A": self._range(2, 4), "B": self._range(4, 8)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["GemmFastGelu"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["GemmFastGelu"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "GemmFastGelu")
        self.assertEqual(node.domain, "com.microsoft")
        self.assertEqual(3, len(node.input))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_gemm_fast_gelu_no_bias(self):
        from yobx.reference import ExtendedReferenceEvaluator

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["A", "B"], ["ab"]),
                    oh.make_node("FastGelu", ["ab"], ["Y"], domain="com.microsoft"),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("A", TFLOAT, [2, 4]),
                    oh.make_tensor_value_info("B", TFLOAT, [4, 8]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 8])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"A": self._range(2, 4), "B": self._range(4, 8)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["GemmFastGelu"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["GemmFastGelu"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(0, len(opt_onx.graph.initializer))
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "GemmFastGelu")
        self.assertEqual(node.domain, "com.microsoft")
        self.assertEqual(2, len(node.input))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_gemm_fast_gelu_fast_gelu_with_bias_input(self):
        from yobx.reference import ExtendedReferenceEvaluator

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["A", "B"], ["ab"]),
                    oh.make_node("FastGelu", ["ab", "bias"], ["Y"], domain="com.microsoft"),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("A", TFLOAT, [2, 4]),
                    oh.make_tensor_value_info("B", TFLOAT, [4, 8]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, [2, 8])],
                [
                    onh.from_array(
                        np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.4, 0.0, -0.3], dtype=np.float32),
                        name="bias",
                    )
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        feeds = {"A": self._range(2, 4), "B": self._range(4, 8)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["GemmFastGelu"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["GemmFastGelu"], [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(1, len(opt_onx.graph.initializer))
        node = opt_onx.graph.node[0]
        self.assertEqual(node.op_type, "GemmFastGelu")
        self.assertEqual(node.domain, "com.microsoft")
        self.assertEqual(3, len(node.input))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_skip_layer_normalization_1d(self):
        from onnxruntime import InferenceSession

        for itype, dtype in [(TFLOAT, np.float32), (TensorProto.FLOAT16, np.float16)]:
            model = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Add", ["X1", "X2"], ["add"]),
                        oh.make_node(
                            "LayerNormalization", ["add", "scale", "bias"], ["Y"], axis=-1
                        ),
                    ],
                    "dummy",
                    [
                        oh.make_tensor_value_info("X1", itype, ["a", "b", "c"]),
                        oh.make_tensor_value_info("X2", itype, ["a", "b", "c"]),
                        oh.make_tensor_value_info("scale", itype, ["c"]),
                        oh.make_tensor_value_info("bias", itype, ["c"]),
                    ],
                    [
                        oh.make_tensor_value_info("add", itype, ["a", "b", "c"]),
                        oh.make_tensor_value_info("Y", itype, ["a", "b", "c"]),
                    ],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
                ir_version=9,
            )
            feeds = {
                "X1": self._range(8, 3, 32).astype(dtype),
                "X2": self._range(8, 3, 32).astype(dtype),
                "scale": self._range(32).astype(dtype),
                "bias": self._range(32).astype(dtype),
            }
            ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            expected = ref.run(None, feeds)

            gr = GraphBuilder(
                model,
                infer_shapes_options=True,
                optimization_options=OptimizationOptions(
                    patterns=["SkipLayerNormalization"], verbose=0
                ),
            )
            opt_onx = gr.to_onnx(optimize=True)
            self.assertIn("SkipLayerNormalization", [n.op_type for n in opt_onx.graph.node])

            opt_ref = InferenceSession(
                opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            got = opt_ref.run(None, feeds)
            self.assertEqualArray(expected[0].ravel(), got[0].ravel())
            self.assertEqualArray(expected[0], got[0])

    def test_skip_layer_normalization_3d(self):
        itype, _dtype = (TFLOAT, np.float32)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X1", "X2"], ["add"]),
                    oh.make_node("LayerNormalization", ["add", "scale", "bias"], ["Y"], axis=-1),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X1", itype, ["a", "b", "c"]),
                    oh.make_tensor_value_info("X2", itype, ["a", "b", "c"]),
                    oh.make_tensor_value_info("scale", itype, ["a", "b", "c"]),
                    oh.make_tensor_value_info("bias", itype, ["a", "b", "c"]),
                ],
                [
                    oh.make_tensor_value_info("add", itype, ["a", "b", "c"]),
                    oh.make_tensor_value_info("Y", itype, ["a", "b", "c"]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["SkipLayerNormalization"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("SkipLayerNormalization", [n.op_type for n in opt_onx.graph.node])

    def test_reshape_gemm(self):
        from onnxruntime import InferenceSession

        for transB in [0, 1]:
            with self.subTest(transB=transB):

                model = oh.make_model(
                    oh.make_graph(
                        [
                            oh.make_node("Reshape", ["A", "shape"], ["xr"]),
                            oh.make_node("Gemm", ["xr", "B"], ["Y"], transB=transB),
                        ],
                        "dummy",
                        [
                            oh.make_tensor_value_info("A", TFLOAT, ["a", "b", 8]),
                            oh.make_tensor_value_info("B", TFLOAT, [4, 8] if transB else [8, 4]),
                        ],
                        [oh.make_tensor_value_info("Y", TFLOAT, ["f", "g"])],
                        [onh.from_array(np.array([-1, 8], dtype=np.int64), name="shape")],
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                    ir_version=9,
                )
                feeds = {
                    "A": self._range(2, 3, 8),
                    "B": self._range(*([4, 8] if transB else [8, 4])),
                }
                ref = InferenceSession(
                    model.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                expected = ref.run(None, feeds)

                gr = GraphBuilder(
                    model,
                    infer_shapes_options=True,
                    optimization_options=OptimizationOptions(patterns=["ReshapeGemm"], verbose=0),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertIn("FusedMatMul", [n.op_type for n in opt_onx.graph.node])

                opt_ref = InferenceSession(
                    opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                got = opt_ref.run(None, feeds)
                self.assertEqualArray(expected[0].ravel(), got[0].ravel())
                self.assertEqualArray(expected[0], got[0])

    def test_reshape_gemm_reshape(self):
        for transB in [0, 1]:
            with self.subTest(transB=transB):
                model = oh.make_model(
                    oh.make_graph(
                        [
                            oh.make_node("Shape", ["A"], ["shapeA"], start=0, end=-1),
                            oh.make_node("Concat", ["shapeA", "m_one"], ["shapey"], axis=0),
                            oh.make_node("Reshape", ["A", "shape"], ["xr"]),
                            oh.make_node("Gemm", ["xr", "B"], ["y2"], transB=transB),
                            oh.make_node("Reshape", ["y2", "shapey"], ["yy"]),
                            oh.make_node("Identity", ["yy"], ["Y"]),
                        ],
                        "dummy",
                        [
                            oh.make_tensor_value_info("A", TFLOAT, ["a", "b", "c"]),
                            oh.make_tensor_value_info("B", TFLOAT, [4, 8] if transB else [8, 4]),
                        ],
                        [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"])],
                        [
                            onh.from_array(np.array([-1, 8], dtype=np.int64), name="shape"),
                            onh.from_array(np.array([-1], dtype=np.int64), name="m_one"),
                        ],
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                    ir_version=9,
                )
                feeds = {
                    "A": self._range(2, 3, 8),
                    "B": self._range(*([4, 8] if transB else [8, 4])),
                }
                ref = self._check_with_ort(model)
                expected = ref.run(None, feeds)

                gr = GraphBuilder(
                    model,
                    infer_shapes_options=True,
                    optimization_options=OptimizationOptions(
                        patterns=["ReshapeGemmReshape"], verbose=0
                    ),
                )
                opt_onx = gr.to_onnx(optimize=True)
                self.assertEqual(["FusedMatMul"], [n.op_type for n in opt_onx.graph.node])

                opt_ref = self._check_with_ort(opt_onx)
                got = opt_ref.run(None, feeds)
                self.assertEqualArray(expected[0].ravel(), got[0].ravel())
                self.assertEqualArray(expected[0], got[0])

    def test_transpose_matmul_b(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["B"], ["xr"], perm=[0, 2, 3, 1]),
                    oh.make_node("MatMul", ["A", "xr"], ["Y"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("A", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("B", TFLOAT, ["i", "j", "k", "l"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["m", "n", "o", "p"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        feeds = {"A": self._range(2, 3, 8, 7), "B": self._range(2, 8, 3, 7)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["TransposeFusedMatMulB"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("FusedMatMul", [n.op_type for n in opt_onx.graph.node])

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0].ravel(), got[0].ravel())
        self.assertEqualArray(expected[0], got[0])

    def test_skip_simplified_layer_normalization(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "skip"], ["xs"]),
                    oh.make_node(
                        "SimplifiedLayerNormalization",
                        ["xs", "scale"],
                        ["Y"],
                        epsilon=1e-1,
                        axis=-1,
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("skip", TFLOAT, ["batch", "cache", 192]),
                ],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("xs", TFLOAT, ["batch", "cache", 192]),
                ],
                [onh.from_array(np.ones(192, dtype=np.float32), name="scale")],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        feeds = {"X": self._range(2, 128, 192), "skip": self._range(2, 128, 192)}
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["SkipSimplifiedLayerNormalization"]
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["SkipSimplifiedLayerNormalization"], [n.op_type for n in opt_onx.graph.node]
        )
        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualAny(expected, got, atol=1e-5)

    def test_skip_simplified_layer_normalization_mul(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "skip"], ["xs"]),
                    oh.make_node(
                        "SimplifiedLayerNormalization",
                        ["xs", "scale"],
                        ["ym"],
                        epsilon=1e-1,
                        axis=-1,
                    ),
                    oh.make_node("Mul", ["ym", "weights"], ["a"]),
                    oh.make_node("Add", ["a", "weights"], ["Y"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("skip", TFLOAT, ["batch", "cache", 192]),
                ],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("xs", TFLOAT, ["batch", "cache", 192]),
                ],
                [
                    onh.from_array(np.ones(192, dtype=np.float32), name="scale"),
                    onh.from_array(self._range(192, bias=1000), name="weights"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["SkipSimplifiedLayerNormalization"]
            ),
        )
        self.dump_onnx("test_skip_simplified_layer_normalization_mul0.onnx", model)
        model2 = gr.to_onnx(optimize=True)
        feeds = {"X": self._range(2, 128, 192, bias=0.001), "skip": self._range(2, 128, 192)}
        self.dump_onnx("test_skip_simplified_layer_normalization_mul1.onnx", model2)

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)
        ref2 = InferenceSession(model2.SerializeToString(), providers=["CPUExecutionProvider"])
        expected2 = ref2.run(None, feeds)
        self.assertEqualAny(expected, expected2, atol=3e-4)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=[
                    "SkipSimplifiedLayerNormalization",
                    "SkipSimplifiedLayerNormalizationMul",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["SkipSimplifiedLayerNormalization", "Add"], [n.op_type for n in opt_onx.graph.node]
        )
        self.dump_onnx("test_skip_simplified_layer_normalization_mul.onnx", opt_onx)
        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)

        self.assertEqualArray(expected2[1], got[1])
        self.assertEqualArray(expected2[0], got[0])

    def test_simplified_layer_normalization_mul(self):
        from onnxruntime import InferenceSession

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "skip"], ["xs"]),
                    oh.make_node(
                        "SimplifiedLayerNormalization",
                        ["xs", "scale"],
                        ["ym"],
                        epsilon=1e-1,
                        axis=-1,
                    ),
                    oh.make_node("Mul", ["ym", "weights"], ["a"]),
                    oh.make_node("Add", ["a", "weights"], ["Y"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("skip", TFLOAT, ["batch", "cache", 192]),
                ],
                [
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "cache", 192]),
                    oh.make_tensor_value_info("xs", TFLOAT, ["batch", "cache", 192]),
                ],
                [
                    onh.from_array(np.ones(192, dtype=np.float32), name="scale"),
                    onh.from_array(self._range(192, bias=1000), name="weights"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        feeds = {"X": self._range(2, 128, 192, bias=0.001), "skip": self._range(2, 128, 192)}
        sess = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = sess.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["SimplifiedLayerNormalizationMul"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Add", "SimplifiedLayerNormalization", "Add"],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.dump_onnx("test_simplified_layer_normalization_mul.onnx", opt_onx)
        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)

        self.assertEqualArray(expected[1], got[1])
        self.assertEqualArray(expected[0], got[0])

    def test_contrib_rotary_embedding_concat_after(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["X", "split"], ["Xh1", "Xh2"], axis=-1),
                    oh.make_node("Split", ["Xh1"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m1x2"], ["cm1"]),
                    oh.make_node("Mul", ["Xh1", "m2x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Yh"]),
                    oh.make_node("Concat", ["Yh", "Xh2"], ["Y"], axis=-1),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", 2, "c", "d"]),
                    oh.make_tensor_value_info("m1", TFLOAT, [1, 1, "c", "e"]),
                    oh.make_tensor_value_info("m2", TFLOAT, [1, 1, "c", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
                [onh.from_array(np.array([4, 6], dtype=np.int64), name="split")],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        shape_x = (2, 2, 3, 10)
        shape_c = (1, 1, 3, 2)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "m1": ((np.arange(np.prod(shape_c)) + 1) / np.prod(shape_c) * 15)
            .reshape(shape_c)
            .astype(np.float32),
            "m2": ((np.arange(np.prod(shape_c)) + 1) / np.prod(shape_c) * 5)
            .reshape(shape_c)
            .astype(np.float32),
        }
        # ExtendedReferenceEvaluator(model, verbose=10).run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["FunctionHalfRotaryEmbedding", "ContribRotaryEmbedding"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_rotary_embedding_concat_after.onnx", opt_onx)
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])

        import onnxruntime

        ref = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        z = ref.run(None, feeds)[0]
        ref = onnxruntime.InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    @hide_stdout()
    def test_contrib_rotary_embedding_no_concat_after(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["X"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m1x2"], ["cm1"]),
                    oh.make_node("Mul", ["X", "m2x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Y"]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", 2, "c", "e*2"]),
                    oh.make_tensor_value_info("m1", TFLOAT, [1, 1, "c", "e"]),
                    oh.make_tensor_value_info("m2", TFLOAT, [1, 1, "c", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "e*2"])],
                [onh.from_array(np.array([4, 6], dtype=np.int64), name="split")],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        shape_x = (2, 2, 3, 16)
        shape_c = (1, 1, 3, 8)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "m1": ((np.arange(np.prod(shape_c)) + 1) / np.prod(shape_c) * 15)
            .reshape(shape_c)
            .astype(np.float32),
            "m2": ((np.arange(np.prod(shape_c)) + 1) / np.prod(shape_c) * 5)
            .reshape(shape_c)
            .astype(np.float32),
        }
        # ExtendedReferenceEvaluator(model, verbose=10).run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["FunctionHalfRotaryEmbedding", "ContribRotaryEmbedding"], verbose=10
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_rotary_embedding_no_concat_after.onnx", opt_onx)
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("Concat", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])

        import onnxruntime

        ref = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        z = ref.run(None, feeds)[0]
        ref = onnxruntime.InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    @hide_stdout()
    def test_contrib_gemma_rotary_embedding(self):
        from onnx.reference.op_run import OpRun

        class GemmaRotaryEmbedding(OpRun):
            op_domain = "com.microsoft"

            def _run(self, emb, q, q_rot, k, k_rot):
                # emb: (batch, seq, dim); q/q_rot/k/k_rot: (batch, heads, seq, dim)
                sin = np.sin(emb)[:, np.newaxis, :, :]
                cos = np.cos(emb)[:, np.newaxis, :, :]
                output1 = q * cos + q_rot * sin
                output2 = k * cos + k_rot * sin
                return (output1.astype(q.dtype), output2.astype(k.dtype))

        opset = 20
        # emb: (batch=2, seq=3, dim=4), q: (batch=2, heads=2, seq=3, dim=4),
        # k: (batch=2, heads=1, seq=3, dim=4)
        model = oh.make_model(
            oh.make_graph(
                [
                    # Compute sin/cos from emb and unsqueeze for broadcasting
                    oh.make_node("Sin", ["emb"], ["sin_emb"]),
                    oh.make_node("Cos", ["emb"], ["cos_emb"]),
                    oh.make_node("Unsqueeze", ["sin_emb", "axes1"], ["sin_4d"]),
                    oh.make_node("Unsqueeze", ["cos_emb", "axes1"], ["cos_4d"]),
                    # q rotary embedding: rotate_half(q) * sin + q * cos
                    oh.make_node("Split", ["q"], ["q1", "q2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["q2"], ["neg_q2"]),
                    oh.make_node("Concat", ["neg_q2", "q1"], ["q_rot"], axis=-1),
                    oh.make_node("Mul", ["q_rot", "sin_4d"], ["q_sin"]),
                    oh.make_node("Mul", ["q", "cos_4d"], ["q_cos"]),
                    oh.make_node("Add", ["q_sin", "q_cos"], ["q_embed"]),
                    # k rotary embedding: rotate_half(k) * sin + k * cos
                    oh.make_node("Split", ["k"], ["k1", "k2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["k2"], ["neg_k2"]),
                    oh.make_node("Concat", ["neg_k2", "k1"], ["k_rot"], axis=-1),
                    oh.make_node("Mul", ["k_rot", "sin_4d"], ["k_sin"]),
                    oh.make_node("Mul", ["k", "cos_4d"], ["k_cos"]),
                    oh.make_node("Add", ["k_sin", "k_cos"], ["k_embed"]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("emb", TFLOAT, [2, 3, 4]),
                    oh.make_tensor_value_info("q", TFLOAT, [2, 2, 3, 4]),
                    oh.make_tensor_value_info("k", TFLOAT, [2, 1, 3, 4]),
                ],
                [
                    oh.make_tensor_value_info("q_embed", TFLOAT, [2, 2, 3, 4]),
                    oh.make_tensor_value_info("k_embed", TFLOAT, [2, 1, 3, 4]),
                ],
                [onh.from_array(np.array([1], dtype=np.int64), name="axes1")],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        rng = np.random.default_rng(42)
        feeds = {
            "emb": rng.random((2, 3, 4)).astype(np.float32),
            "q": rng.random((2, 2, 3, 4)).astype(np.float32),
            "k": rng.random((2, 1, 3, 4)).astype(np.float32),
        }

        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["FunctionHalfRotaryEmbedding", "ContribGemmaRotaryEmbedding"],
                verbose=10,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_gemma_rotary_embedding.onnx", opt_onx)
        self.assertIn("GemmaRotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])

        # Verify the output of the fused graph matches using a custom reference evaluator
        ref_opt = ExtendedReferenceEvaluator(opt_onx, new_ops=[GemmaRotaryEmbedding])
        got = ref_opt.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)
        self.assertEqualArray(expected[1], got[1], atol=1e-5)

    @hide_stdout()
    def test_missing_kernels(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["X"], ["xc"], to=TensorProto.BFLOAT16),
                    oh.make_node("Sin", ["xc"], ["xcs"]),
                    oh.make_node("Cos", ["xc"], ["xcc"]),
                    oh.make_node("Cast", ["zero"], ["zeroc"], to=TensorProto.BFLOAT16),
                    oh.make_node("Cast", ["one"], ["onec"], to=TensorProto.BFLOAT16),
                    oh.make_node("Size", ["X"], ["size"]),
                    oh.make_node("Cast", ["size"], ["sizec"], to=TensorProto.BFLOAT16),
                    oh.make_node("Range", ["zeroc", "onec", "sizec"], ["y"]),
                    oh.make_node("Add", ["xcc", "xcs"], ["xccc"]),
                    oh.make_node("Add", ["xccc", "y"], ["yc"]),
                    oh.make_node("Cast", ["yc"], ["Y"], to=TensorProto.FLOAT),
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, ["a"])],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a"])],
                [
                    onh.from_array(np.array(0, dtype=np.int64), name="zero"),
                    onh.from_array(np.array(1, dtype=np.int64), name="one"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )
        self.assertIn("Range", [n.op_type for n in model.graph.node])
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["MissingRange", "MissingCosSin"], verbose=10
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("Range", [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(
            [
                "Size",
                "Cast",
                "Cast",
                "Sin",
                "Cast",
                "Cast",
                "Cos",
                "Cast",
                "Cast",
                "Cast",
                "Range",
                "Cast",
                "Add",
                "Add",
                "Cast",
            ],
            [n.op_type for n in opt_onx.graph.node],
        )

        import onnxruntime

        # feeds = {"X": np.arange(5).astype(np.float32)}
        self.assertRaise(
            lambda: self.make_inference_session(model),
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph,
        )

    def test_contrib_rotary_embedding_concat_after_position_ids(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["position_ids"], ["batch"], start=0, end=1),
                    oh.make_node("Concat", ["batch", "init11"], ["new_shape"], axis=0),
                    oh.make_node("Unsqueeze", ["weights", "init01"], ["weights_u"]),
                    oh.make_node("Expand", ["weights_u", "new_shape"], ["weights_expanded"]),
                    oh.make_node("Unsqueeze", ["position_ids", "one"], ["pids1"]),
                    oh.make_node("Cast", ["pids1"], ["cids"], to=TensorProto.FLOAT),
                    oh.make_node("Reshape", ["cids", "init0_11"], ["resh"]),
                    oh.make_node("Mul", ["weights_expanded", "resh"], ["milti"]),
                    oh.make_node("Cos", ["milti"], ["m1s"]),
                    oh.make_node("Sin", ["milti"], ["m2s"]),
                    oh.make_node("Unsqueeze", ["m1s", "one"], ["m1"]),
                    oh.make_node("Unsqueeze", ["m2s", "one"], ["m2"]),
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["X", "split"], ["Xh1", "Xh2"], axis=-1),
                    oh.make_node("Split", ["Xh1"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m2x2"], ["cm1"]),
                    oh.make_node("Mul", ["Xh1", "m1x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Yh"]),
                    oh.make_node("Concat", ["Yh", "Xh2"], ["Y"], axis=-1),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", 2, "c", "d"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["a", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
                [
                    onh.from_array(np.array([4, 6], dtype=np.int64), name="split"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                    onh.from_array(np.array([0, -1, 1], dtype=np.int64), name="init0_11"),
                    onh.from_array(np.array(1, dtype=np.int64), name="one_no_dim"),
                    onh.from_array(np.array([0.1, 0.2], dtype=np.float32), name="weights"),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="init01"),
                    onh.from_array(np.array([1, 1], dtype=np.int64), name="init11"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        shape_x = (2, 2, 3, 10)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "position_ids": np.array([[1, 3, 6], [2, 4, 7]], dtype=np.int64),
        }
        ExtendedReferenceEvaluator(model, verbose=0).run(None, feeds)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["FunctionCosSinCache", "FunctionHalfRotaryEmbedding"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_rotary_embedding_concat_after_position_ids0.onnx", opt_onx)
        ref = self.make_inference_session(model)
        z = ref.run(None, feeds)[0]
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                    "ContribRotaryEmbedding",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_rotary_embedding_concat_after_position_ids.onnx", opt_onx)
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    def test_contrib_rotary_embedding_concat_after_position_ids_3d(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["Xt"], perm=[0, 2, 1, 3]),
                    oh.make_node("Shape", ["position_ids"], ["batch"], start=0, end=1),
                    oh.make_node("Concat", ["batch", "init11"], ["new_shape"], axis=0),
                    oh.make_node("Unsqueeze", ["weights", "init01"], ["weights_u"]),
                    oh.make_node("Expand", ["weights_u", "new_shape"], ["weights_expanded"]),
                    oh.make_node("Unsqueeze", ["position_ids", "one"], ["pids1"]),
                    oh.make_node("Cast", ["pids1"], ["cids"], to=TensorProto.FLOAT),
                    oh.make_node("Reshape", ["cids", "init0_11"], ["resh"]),
                    oh.make_node("Mul", ["weights_expanded", "resh"], ["milti"]),
                    oh.make_node("Cos", ["milti"], ["m1s"]),
                    oh.make_node("Sin", ["milti"], ["m2s"]),
                    oh.make_node("Unsqueeze", ["m1s", "one"], ["m1"]),
                    oh.make_node("Unsqueeze", ["m2s", "one"], ["m2"]),
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["Xt", "split"], ["Xh1", "Xh2"], axis=-1),
                    oh.make_node("Split", ["Xh1"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m2x2"], ["cm1"]),
                    oh.make_node("Mul", ["Xh1", "m1x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Yh"]),
                    oh.make_node("Concat", ["Yh", "Xh2"], ["Y"], axis=-1),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "c", 2, "d"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["a", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
                [
                    onh.from_array(np.array([4, 6], dtype=np.int64), name="split"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                    onh.from_array(np.array([0, -1, 1], dtype=np.int64), name="init0_11"),
                    onh.from_array(np.array(1, dtype=np.int64), name="one_no_dim"),
                    onh.from_array(np.array([0.1, 0.2], dtype=np.float32), name="weights"),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="init01"),
                    onh.from_array(np.array([1, 1], dtype=np.int64), name="init11"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        shape_x = (2, 3, 2, 10)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "position_ids": np.array([[1, 3, 6], [2, 4, 7]], dtype=np.int64),
        }
        ExtendedReferenceEvaluator(model, verbose=0).run(None, feeds)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                    "ContribRotaryEmbedding",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx(
            "test_contrib_rotary_embedding_concat_after_position_ids_3d0.onnx", opt_onx
        )
        ref = self.make_inference_session(model)
        z = ref.run(None, feeds)[0]
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                    "ContribRotaryEmbedding",
                    "ContribRotaryEmbedding3D",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_contrib_rotary_embedding_concat_after_position_ids_3d.onnx", opt_onx)
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertEqual(["Transpose"], [n.op_type for n in opt_onx.graph.node][-1:])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    def test_contrib_rotary_embedding_concat_after_position_ids_swap(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["position_ids"], ["batch"], start=0, end=1),
                    oh.make_node("Concat", ["batch", "init11"], ["new_shape"], axis=0),
                    oh.make_node("Unsqueeze", ["weights", "init02"], ["weights_u"]),
                    oh.make_node("Expand", ["weights_u", "new_shape"], ["weights_expandede"]),
                    oh.make_node(
                        "Reshape", ["weights_expandede", "init01_1"], ["weights_expanded"]
                    ),
                    oh.make_node("Unsqueeze", ["position_ids", "one"], ["pids1"]),
                    oh.make_node("Cast", ["pids1"], ["cids"], to=TensorProto.FLOAT),
                    oh.make_node("Reshape", ["cids", "init0_11"], ["resh"]),
                    oh.make_node("Mul", ["weights_expanded", "resh"], ["milti"]),
                    oh.make_node("Cos", ["milti"], ["m1s"]),
                    oh.make_node("Sin", ["milti"], ["m2s"]),
                    oh.make_node("Unsqueeze", ["m1s", "one"], ["m1"]),
                    oh.make_node("Unsqueeze", ["m2s", "one"], ["m2"]),
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["X", "split"], ["Xh1", "Xh2"], axis=-1),
                    oh.make_node("Split", ["Xh1"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m2x2"], ["cm1"]),
                    oh.make_node("Mul", ["Xh1", "m1x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Yh"]),
                    oh.make_node("Concat", ["Yh", "Xh2"], ["Y"], axis=-1),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", 2, "c", "d"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["a", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
                [
                    onh.from_array(np.array([4, 6], dtype=np.int64), name="split"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                    onh.from_array(np.array([0, -1, 1], dtype=np.int64), name="init0_11"),
                    onh.from_array(np.array([0, 1, -1], dtype=np.int64), name="init01_1"),
                    onh.from_array(np.array(1, dtype=np.int64), name="one_no_dim"),
                    onh.from_array(np.array([0.1, 0.2], dtype=np.float32), name="weights"),
                    onh.from_array(np.array([0, 2], dtype=np.int64), name="init02"),
                    onh.from_array(np.array([1, 1], dtype=np.int64), name="init11"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )

        shape_x = (2, 2, 3, 10)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "position_ids": np.array([[1, 3, 6], [2, 4, 7]], dtype=np.int64),
        }
        ExtendedReferenceEvaluator(model, verbose=0).run(None, feeds)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "SwapExpandReshape",
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx(
            "test_contrib_rotary_embedding_concat_after_position_ids_swap0.onnx", opt_onx
        )
        ref = self.make_inference_session(model)
        z = ref.run(None, feeds)[0]
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "SwapExpandReshape",
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                    "ContribRotaryEmbedding",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx(
            "test_contrib_rotary_embedding_concat_after_position_ids_swap.onnx", opt_onx
        )
        self.assertIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])
        sess = self.make_inference_session(opt_onx)
        zz = sess.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-4)

    def test_contrib_rotary_embedding_concat_after_position_ids_no_match(self):
        opset = 20
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["position_ids"], ["batch"], start=0, end=1),
                    oh.make_node("Concat", ["batch", "init11"], ["new_shape"], axis=0),
                    oh.make_node("Unsqueeze", ["weights", "init01"], ["weights_u"]),
                    oh.make_node("Expand", ["weights_u", "new_shape"], ["weights_expanded"]),
                    oh.make_node("Unsqueeze", ["position_ids", "one"], ["pids1"]),
                    oh.make_node("Cast", ["pids1"], ["cids"], to=TensorProto.FLOAT),
                    oh.make_node("Reshape", ["cids", "init0_11"], ["resh"]),
                    oh.make_node("Mul", ["weights_expanded", "resh"], ["milti"]),
                    oh.make_node("Cos", ["milti"], ["m1s"]),
                    oh.make_node("Sin", ["milti"], ["m2s"]),
                    oh.make_node("Unsqueeze", ["m1s", "one"], ["m1"]),
                    oh.make_node("Unsqueeze", ["m2s", "one"], ["m2"]),
                    oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1),
                    oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1),
                    oh.make_node("Split", ["X", "split"], ["Xh1", "Xh2"], axis=-1),
                    oh.make_node("Split", ["Xh1"], ["x1", "x2"], axis=-1, num_outputs=2),
                    oh.make_node("Neg", ["x2"], ["nx2"]),
                    oh.make_node("Concat", ["nx2", "x1"], ["cc"], axis=-1),
                    oh.make_node("Mul", ["cc", "m1x2"], ["cm1"]),
                    oh.make_node("Mul", ["Xh1", "m2x2"], ["cm2"]),
                    oh.make_node("Add", ["cm1", "cm2"], ["Yh"]),
                    oh.make_node("Concat", ["Yh", "Xh2"], ["Y"], axis=-1),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", 2, "c", "d"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["a", "e"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"])],
                [
                    onh.from_array(np.array([4, 6], dtype=np.int64), name="split"),
                    onh.from_array(np.array([1], dtype=np.int64), name="one"),
                    onh.from_array(np.array([0, -1, 1], dtype=np.int64), name="init0_11"),
                    onh.from_array(np.array(1, dtype=np.int64), name="one_no_dim"),
                    onh.from_array(np.array([0.1, 0.2], dtype=np.float32), name="weights"),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="init01"),
                    onh.from_array(np.array([1, 1], dtype=np.int64), name="init11"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", opset)],
            ir_version=10,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=["FunctionCosSinCache", "FunctionHalfRotaryEmbedding"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx(
            "test_contrib_rotary_embedding_concat_after_position_ids_no_match.onnx", opt_onx
        )

        shape_x = (2, 2, 3, 10)
        feeds = {
            "X": ((np.arange(np.prod(shape_x)) + 1) / (np.prod(shape_x) * 10))
            .reshape(shape_x)
            .astype(np.float32),
            "position_ids": np.array([[1, 3, 6], [2, 4, 7]], dtype=np.int64),
        }
        ExtendedReferenceEvaluator(model, verbose=0).run(None, feeds)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(
                patterns=[
                    "FunctionCosSinCache",
                    "FunctionHalfRotaryEmbedding",
                    "ContribRotaryEmbedding",
                ],
                verbose=0,
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("RotaryEmbedding", [n.op_type for n in opt_onx.graph.node])

    def test_multi_head_attention(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["query"], ["t_query"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["keys"], ["t_keys"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["values"], ["t_values"], perm=[0, 2, 1, 3]),
                    oh.make_node("Concat", ["past_keys", "t_keys"], ["ct_keys"], axis=-2),
                    oh.make_node("Concat", ["past_values", "t_values"], ["ct_values"], axis=-2),
                    oh.make_node("Mul", ["t_query", "scale_sqrt"], ["query_scaled"]),
                    oh.make_node("Mul", ["ct_keys", "scale_sqrt"], ["keys_scaled"]),
                    oh.make_node(
                        "Transpose", ["keys_scaled"], ["keys_scaled_t"], perm=[0, 1, 3, 2]
                    ),
                    oh.make_node("MatMul", ["query_scaled", "keys_scaled_t"], ["qk"]),
                    oh.make_node("Where", ["mask", "zero", "minfty"], ["bias"]),
                    oh.make_node("Add", ["qk", "bias"], ["qkb"]),
                    oh.make_node("Softmax", ["qkb"], ["qkbs"], axis=-1),
                    oh.make_node("IsNaN", ["qkbs"], ["nans"]),
                    oh.make_node("Where", ["nans", "zero", "qkbs"], ["filt"]),
                    oh.make_node("MatMul", ["filt", "ct_values"], ["prob"]),
                    oh.make_node("Transpose", ["prob"], ["Y"], perm=[0, 2, 1, 3]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("query", TFLOAT, ["aq", "bq", 8, 64]),
                    oh.make_tensor_value_info("keys", TFLOAT, ["ak", "bk", 8, 64]),
                    oh.make_tensor_value_info("values", TFLOAT, ["av", "bv", 8, 64]),
                    oh.make_tensor_value_info("past_keys", TFLOAT, ["pak", 8, "pck", 64]),
                    oh.make_tensor_value_info("past_values", TFLOAT, ["pav", 8, "pcv", 64]),
                    oh.make_tensor_value_info("mask", TensorProto.BOOL, ["am", 1, "cm", "dm"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["ay", "by", "cy", "dy"])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([-np.inf], dtype=np.float32), name="minfty"),
                    onh.from_array(np.array([0.1**0.5], dtype=np.float32), name="scale_sqrt"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        feeds = dict(
            query=np.random.randn(32, 128, 8, 64).astype(np.float32),
            keys=np.random.randn(32, 128, 8, 64).astype(np.float32),
            values=np.random.randn(32, 128, 8, 64).astype(np.float32),
            mask=np.random.rand(32, 1, 128, 256) >= 0.5,
            past_keys=np.random.randn(32, 8, 128, 64).astype(np.float32),
            past_values=np.random.randn(32, 8, 128, 64).astype(np.float32),
        )
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        z = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FunctionAttention", "MultiHeadAttention3D"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_multi_head_attention.onnx", opt_onx)
        ref = self.make_inference_session(opt_onx)
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-3)
        self.assertEqual(
            ["Reshape", "Reshape", "Reshape", "Where", "MultiHeadAttention", "Reshape"],
            [n.op_type for n in opt_onx.graph.node],
        )

    def test_multi_head_attention_where_add(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["query"], ["t_query"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["keys"], ["t_keys"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["values"], ["t_values"], perm=[0, 2, 1, 3]),
                    oh.make_node("Concat", ["past_keys", "t_keys"], ["ct_keys"], axis=-2),
                    oh.make_node("Concat", ["past_values", "t_values"], ["ct_values"], axis=-2),
                    oh.make_node("Mul", ["t_query", "scale_sqrt"], ["query_scaled"]),
                    oh.make_node("Mul", ["ct_keys", "scale_sqrt"], ["keys_scaled"]),
                    oh.make_node(
                        "Transpose", ["keys_scaled"], ["keys_scaled_t"], perm=[0, 1, 3, 2]
                    ),
                    oh.make_node("MatMul", ["query_scaled", "keys_scaled_t"], ["qk"]),
                    oh.make_node("Where", ["mask", "zero", "minfty"], ["bias"]),
                    oh.make_node("Add", ["qk", "bias"], ["qkb"]),
                    oh.make_node("Softmax", ["qkb"], ["qkbs"], axis=-1),
                    oh.make_node("IsNaN", ["qkbs"], ["nans"]),
                    oh.make_node("Where", ["nans", "zero", "qkbs"], ["filt"]),
                    oh.make_node("MatMul", ["filt", "ct_values"], ["prob"]),
                    oh.make_node("Transpose", ["prob"], ["Y"], perm=[0, 2, 1, 3]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("query", TFLOAT, ["aq", "bq", 8, 64]),
                    oh.make_tensor_value_info("keys", TFLOAT, ["ak", "bk", 8, 64]),
                    oh.make_tensor_value_info("values", TFLOAT, ["av", "bv", 8, 64]),
                    oh.make_tensor_value_info("past_keys", TFLOAT, ["pak", 8, "pck", 64]),
                    oh.make_tensor_value_info("past_values", TFLOAT, ["pav", 8, "pcv", 64]),
                    oh.make_tensor_value_info("mask", TensorProto.BOOL, ["am", 1, "cm", "dm"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["ay", "by", "cy", "dy"])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([-np.inf], dtype=np.float32), name="minfty"),
                    onh.from_array(np.array([0.1**0.5], dtype=np.float32), name="scale_sqrt"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 18)],
            ir_version=10,
        )
        feeds = dict(
            query=np.random.randn(32, 128, 8, 64).astype(np.float32),
            keys=np.random.randn(32, 128, 8, 64).astype(np.float32),
            values=np.random.randn(32, 128, 8, 64).astype(np.float32),
            mask=np.random.rand(32, 1, 128, 256) >= 0.5,
            past_keys=np.random.randn(32, 8, 128, 64).astype(np.float32),
            past_values=np.random.randn(32, 8, 128, 64).astype(np.float32),
        )
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        z = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FunctionAttention", "MultiHeadAttention3D"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_multi_head_attention.onnx", opt_onx)
        self.assertEqual(
            ["Reshape", "Reshape", "Reshape", "Where", "MultiHeadAttention", "Reshape"],
            [n.op_type for n in opt_onx.graph.node],
        )
        ref = self.make_inference_session(opt_onx)
        zz = ref.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-3)

    def _get_model_attention(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(
                self,
                sequence_length,
                num_heads,
                kv_num_heads,
                head_size,
                softmax_scale,
                use_smooth_softmax,
            ):
                super().__init__()
                self.sequence_length = sequence_length
                self.num_heads = num_heads
                self.kv_num_heads = kv_num_heads
                self.head_size = head_size
                self.softmax_scale = softmax_scale
                self.use_smooth_softmax = use_smooth_softmax

            def concat_cache(self, past_key_cache, new_key):
                assert past_key_cache.size(0) == new_key.size(
                    0
                ), f"Batch sizes do not match, {past_key_cache.shape=}, {new_key.shape=}"
                assert past_key_cache.size(1) == new_key.size(
                    1
                ), f"Number of heads do not match, {past_key_cache.shape=}, {new_key.shape=}"
                assert past_key_cache.size(3) == new_key.size(
                    3
                ), f"Head dimensions do not match, {past_key_cache.shape=}, {new_key.shape=}"
                concatenated_keys = torch.cat((past_key_cache, new_key), dim=2)
                return concatenated_keys

            def smooth_softmax_ref(self, x, head_sink):
                assert len(x.shape) == 4
                b, n, s, _t = x.shape

                if head_sink is not None:
                    assert len(head_sink.shape) == 1
                    assert head_sink.shape[0] == x.shape[1]
                    sink = head_sink.reshape(1, n, 1, 1).expand(b, -1, s, -1)
                else:
                    sink = torch.zeros(b, n, s, 1, dtype=x.dtype)

                y = torch.cat([x, sink], dim=-1)
                y = torch.softmax(y, dim=-1)
                y = y[..., :-1]
                return y

            def group_query_attention_reference(self, query, key, value, scale=None, mask=None):
                if scale is None:
                    scale = 1.0 / (self.head_size**0.5)

                num_key_value_groups = self.num_heads // self.kv_num_heads
                value = torch.repeat_interleave(value, dim=1, repeats=num_key_value_groups)
                key = torch.repeat_interleave(key, dim=1, repeats=num_key_value_groups)
                # attn = torch.einsum("bhmd,bhnd->bhmn", query, key).float() * scale
                attn = torch.matmul(query * scale**0.5, key.transpose(2, 3) * scale**0.5)
                if mask is not None:
                    attn = attn.masked_fill(~mask, float("-inf")).to(query.dtype)
                # the exporter does not like this.
                # torch._check(attn.max().item() > -10000, lambda: "mask is only False")

                attn = (
                    self.smooth_softmax_ref(attn, None)
                    if self.use_smooth_softmax
                    else attn.softmax(-1)
                )
                attn = torch.where(
                    attn.isnan(), torch.tensor(0, dtype=query.dtype, device=query.device), attn
                )
                return torch.matmul(attn, value)

            def forward(self, query, key, value, attention_mask, past_key, past_value):
                present_key = self.concat_cache(past_key, key)
                present_value = self.concat_cache(past_value, value)
                return self.group_query_attention_reference(
                    query,
                    present_key,
                    present_value,
                    scale=self.softmax_scale,
                    mask=attention_mask,
                )

        model = Model(
            sequence_length=1,
            num_heads=8,
            kv_num_heads=4,
            head_size=32,
            softmax_scale=None,
            use_smooth_softmax=False,
        ).eval()

        past_length = 22
        query = torch.rand((1, model.num_heads, model.sequence_length, model.head_size))
        key = torch.rand((1, model.kv_num_heads, model.sequence_length, model.head_size))
        value = torch.rand((1, model.kv_num_heads, model.sequence_length, model.head_size))
        past_key = torch.rand((1, model.kv_num_heads, past_length, model.head_size))
        past_value = torch.rand((1, model.kv_num_heads, past_length, model.head_size))
        attention_mask = torch.randint(
            0, 2, (model.sequence_length, model.sequence_length + past_length)
        ).to(bool)

        inputs = (query, key, value, attention_mask, past_key, past_value)
        expected = model.forward(*inputs)
        self.assertEqual((1, 8, model.sequence_length, 32), expected.shape)
        ds = (
            {0: "batch", 2: "seq_length"},
            {0: "batch", 2: "seq_length"},
            {0: "batch", 2: "seq_length"},
            {0: "seq_length", 1: "total_length"},
            {0: "batch", 2: "past_length"},
            {0: "batch", 2: "past_length"},
        )
        return model, inputs, ds, expected

    @requires_torch()
    @ignore_warnings((UserWarning, FutureWarning))
    @hide_stdout()
    def test_gqa_default(self):
        from yobx.torch import to_onnx

        model, inputs, ds, expected = self._get_model_attention()
        f1 = self.get_dump_file("test_gqa.default.default.custom.onnx")
        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=f1,
            options=OptimizationOptions(patterns="default"),
        )
        self.assertIn("LocalAttentionGQA_to1", [f.op_type for f in onx.graph.node])
        ort = self._check_with_ort(onx, cpu=True)
        feeds = dict(zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in inputs]))
        got = ort.run(None, feeds)
        self.assertEqualArray(expected, got[0], 1e-5)

    @requires_torch()
    @ignore_warnings((UserWarning, FutureWarning))
    @hide_stdout()
    def test_gqa_ort_contribops(self):
        from yobx.torch import to_onnx

        model, inputs, ds, expected = self._get_model_attention()
        f1 = self.get_dump_file("test_gqa_ort_contribops.onnx")
        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=f1,
            options=OptimizationOptions(patterns="default+onnxruntime"),
            target_opset=22,
        )
        self.assertIn("GroupQueryAttention", [f.op_type for f in onx.graph.node])
        ort = self._check_with_ort(onx, cpu=True)
        feeds = dict(zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in inputs]))
        got = ort.run(None, feeds)
        self.assertEqualArray(expected, got[0], 1e-5)

    @requires_torch()
    @ignore_warnings((UserWarning, FutureWarning))
    @hide_stdout()
    def test_gqa_ort_attention(self):
        from yobx.torch import to_onnx

        model, inputs, ds, expected = self._get_model_attention()
        f1 = self.get_dump_file("test_gqa_ort_24.onnx")
        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=f1,
            options=OptimizationOptions(patterns="default+onnxruntime"),
            target_opset=24,
        )
        self.assertIn("Attention", [f.op_type for f in onx.graph.node])
        ort = self._check_with_ort(onx, cpu=True)
        feeds = dict(zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in inputs]))
        got = ort.run(None, feeds)
        self.assertEqualArray(expected, got[0], 1e-5)

    @requires_onnxruntime("1.24")
    def test_onnx_gqa_no_rotary_4D(self):
        _mkv_ = oh.make_tensor_value_info

        num_heads = 8
        kv_num_heads = 4
        head_size = 32
        sequence_length = 1
        past_length = 22
        scale = 0.43 / head_size**0.5

        query = np.random.rand(*(1, num_heads, sequence_length, head_size))
        key = np.random.rand(*(1, kv_num_heads, sequence_length, head_size))
        value = np.random.rand(*(1, kv_num_heads, sequence_length, head_size))
        past_key = np.random.rand(*(1, kv_num_heads, past_length, head_size))
        past_value = np.random.rand(*(1, kv_num_heads, past_length, head_size))
        attention_mask = np.random.randint(
            0, 1, size=(sequence_length, sequence_length + past_length)
        ).astype(bool)

        inputs = (
            query.astype(np.float32),
            key.astype(np.float32),
            value.astype(np.float32),
            attention_mask,
            past_key.astype(np.float32),
            past_value.astype(np.float32),
        )

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Attention",
                        ["query", "key", "value", "mask", "past_key", "past_value"],
                        ["attn", "present_key", "present_value"],
                        scale=scale,
                    ),
                    # QGA contribops
                    oh.make_node("Shape", ["query"], ["batch"], end=1),
                    oh.make_node("Where", ["mask", "zero", "infty"], ["float_mask"]),
                    oh.make_node("Unsqueeze", ["float_mask", "cst01"], ["expanded_mask"]),
                    oh.make_node("Shape", ["mask"], ["total_seqlength64"], start=-1),
                    oh.make_node(
                        "Cast", ["total_seqlength64"], ["total_seqlength"], to=TensorProto.INT32
                    ),
                    oh.make_node("Sub", ["total_seqlength", "one"], ["total_seqlength_1"]),
                    oh.make_node("Expand", ["total_seqlength_1", "batch"], ["seqlensk"]),
                    oh.make_node("Transpose", ["query"], ["query4D"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["key"], ["keys4D"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["value"], ["values4D"], perm=[0, 2, 1, 3]),
                    oh.make_node("Reshape", ["query4D", "shape00"], ["query3D"]),
                    oh.make_node("Reshape", ["keys4D", "shape00"], ["keys3D"]),
                    oh.make_node("Reshape", ["values4D", "shape00"], ["values3D"]),
                    oh.make_node(
                        "GroupQueryAttention",
                        [
                            "query3D",
                            "keys3D",
                            "values3D",
                            "past_key",
                            "past_value",
                            "seqlensk",
                            "total_seqlength",
                            "",
                            "",
                            "",
                            "expanded_mask",
                        ],
                        ["attn3D", "present_key_gqa", "present_value_gqa"],
                        do_rotary=0,
                        num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                        rotary_interleaved=0,
                        scale=scale,
                        domain="com.microsoft",
                    ),
                    oh.make_node("Reshape", ["attn3D", "shape0000"], ["attn4D"]),
                    oh.make_node("Transpose", ["attn4D"], ["attn_gqa"], perm=[0, 2, 1, 3]),
                ],
                "gqa",
                [
                    _mkv_("query", TFLOAT, ["b", "h", "l", "s"]),
                    _mkv_("key", TFLOAT, ["b", "h2", "l2", "s"]),
                    _mkv_("value", TFLOAT, ["b", "h2", "l2", "s"]),
                    _mkv_("mask", TensorProto.BOOL, ["m1", "m2"]),
                    _mkv_("past_key", TFLOAT, ["b", "h3", "lp", "s"]),
                    _mkv_("past_value", TFLOAT, ["b", "h3", "lp", "s"]),
                ],
                [
                    _mkv_("attn", TFLOAT, ["b", "h3", "l3", "s"]),
                    _mkv_("present_key", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("present_value", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("attn_gqa", TFLOAT, ["b", "h3", "l3", "s"]),
                    _mkv_("present_key_gqa", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("present_value_gqa", TFLOAT, ["b", "ho", "lo", "s"]),
                ],
                [
                    onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="shape00"),
                    onh.from_array(
                        np.array([0, 0, -1, head_size], dtype=np.int64), name="shape0000"
                    ),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="cst01"),
                    onh.from_array(np.array([1], dtype=np.int32), name="one"),
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(
                        np.array([np.finfo(np.float32).min], dtype=np.float32), name="infty"
                    ),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 24), oh.make_opsetid("com.microsoft", 1)],
            ir_version=11,
        )
        model = shape_inference.infer_shapes(model)
        check_model(model)
        self.dump_onnx("test_onnx_gqa_no_rotary_4D.onnx", model)

        sess = self._check_with_ort(model, cpu=True)
        feeds = dict(zip([i.name for i in model.graph.input], inputs))
        got = sess.run(None, feeds)
        self.assertEqualArray(got[1], got[4], atol=1e-5)
        self.assertEqualArray(got[2], got[5], atol=1e-5)
        self.assertEqualArray(got[0], got[3], atol=1e-5)

    @requires_onnxruntime("1.24")
    def test_onnx_gqa_no_rotary_3D(self):
        _mkv_ = oh.make_tensor_value_info

        num_heads = 8
        kv_num_heads = 4
        head_size = 32
        sequence_length = 1
        past_length = 22
        scale = 0.43 / head_size**0.5

        query = np.random.rand(*(1, sequence_length, num_heads * head_size))
        key = np.random.rand(*(1, sequence_length, kv_num_heads * head_size))
        value = np.random.rand(*(1, sequence_length, kv_num_heads * head_size))
        past_key = np.random.rand(*(1, kv_num_heads, past_length, head_size))
        past_value = np.random.rand(*(1, kv_num_heads, past_length, head_size))
        attention_mask = np.random.randint(
            0, 1, size=(sequence_length, sequence_length + past_length)
        ).astype(bool)

        inputs = (
            query.astype(np.float32),
            key.astype(np.float32),
            value.astype(np.float32),
            attention_mask,
            past_key.astype(np.float32),
            past_value.astype(np.float32),
        )

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Attention",
                        ["query", "key", "value", "mask", "past_key", "past_value"],
                        ["attn", "present_key", "present_value"],
                        scale=scale,
                        q_num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                    ),
                    # QGA contribops
                    oh.make_node("Shape", ["query"], ["batch"], end=1),
                    oh.make_node("Where", ["mask", "zero", "infty"], ["float_mask"]),
                    oh.make_node("Unsqueeze", ["float_mask", "cst01"], ["expanded_mask"]),
                    oh.make_node("Shape", ["mask"], ["total_seqlength64"], start=-1),
                    oh.make_node(
                        "Cast", ["total_seqlength64"], ["total_seqlength"], to=TensorProto.INT32
                    ),
                    oh.make_node("Sub", ["total_seqlength", "one"], ["total_seqlength_1"]),
                    oh.make_node("Expand", ["total_seqlength_1", "batch"], ["seqlensk"]),
                    oh.make_node(
                        "GroupQueryAttention",
                        [
                            "query",
                            "key",
                            "value",
                            "past_key",
                            "past_value",
                            "seqlensk",
                            "total_seqlength",
                            "",
                            "",
                            "",
                            "expanded_mask",
                        ],
                        ["attn_gqa", "present_key_gqa", "present_value_gqa"],
                        do_rotary=0,
                        num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                        rotary_interleaved=0,
                        scale=scale,
                        domain="com.microsoft",
                    ),
                ],
                "gqa",
                [
                    _mkv_("query", TFLOAT, ["b", "l", "hs"]),
                    _mkv_("key", TFLOAT, ["b", "l2", "h2s"]),
                    _mkv_("value", TFLOAT, ["b", "l2", "h2s"]),
                    _mkv_("mask", TensorProto.BOOL, ["m1", "m2"]),
                    _mkv_("past_key", TFLOAT, ["b", "h3", "lp", "s"]),
                    _mkv_("past_value", TFLOAT, ["b", "h3", "lp", "s"]),
                ],
                [
                    _mkv_("attn", TFLOAT, ["b", "l3", "h3s"]),
                    _mkv_("present_key", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("present_value", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("attn_gqa", TFLOAT, ["b", "l3", "h3s"]),
                    _mkv_("present_key_gqa", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("present_value_gqa", TFLOAT, ["b", "ho", "lo", "s"]),
                ],
                [
                    # onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="shape00"),
                    # onh.from_array(
                    #    np.array([0, 0, -1, head_size], dtype=np.int64), name="shape0000"
                    # ),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="cst01"),
                    onh.from_array(np.array([1], dtype=np.int32), name="one"),
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(
                        np.array([np.finfo(np.float32).min], dtype=np.float32), name="infty"
                    ),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 24), oh.make_opsetid("com.microsoft", 1)],
            ir_version=11,
        )
        model = shape_inference.infer_shapes(model)
        check_model(model)
        self.dump_onnx("test_onnx_gqa_no_rotary_3D.onnx", model)

        sess = self._check_with_ort(model, cpu=True)
        feeds = dict(zip([i.name for i in model.graph.input], inputs))
        got = sess.run(None, feeds)
        self.assertEqualArray(got[1], got[4], atol=1e-5)
        self.assertEqualArray(got[2], got[5], atol=1e-5)
        self.assertEqualArray(got[0], got[3], atol=1e-5)

    @requires_onnxruntime("1.24")
    def test_onnx_gqa_no_rotary_packed_3D(self):
        _mkv_ = oh.make_tensor_value_info

        num_heads = 8
        kv_num_heads = 4
        head_size = 32
        sequence_length = 1
        past_length = 22
        scale = 0.43 / head_size**0.5

        query = np.random.rand(*(1, sequence_length, num_heads * head_size))
        key = np.random.rand(*(1, sequence_length, kv_num_heads * head_size))
        value = np.random.rand(*(1, sequence_length, kv_num_heads * head_size))
        past_key = np.random.rand(*(1, kv_num_heads, past_length, head_size))
        past_value = np.random.rand(*(1, kv_num_heads, past_length, head_size))
        attention_mask = np.random.randint(
            0, 1, size=(sequence_length, sequence_length + past_length)
        ).astype(bool)

        # something is wrong here
        # query[:,:,:,:] = 1
        # key[:,:,:,:] = 1
        # value[:, :, :] = 1
        # value[-1,-1,-1,-1] = 0
        # past_key[:,:,:,:] = 1
        # past_value[:, :, :, :] = 1
        # attention_mask[:,:] = False

        inputs = (
            query.astype(np.float32),
            key.astype(np.float32),
            value.astype(np.float32),
            attention_mask,
            past_key.astype(np.float32),
            past_value.astype(np.float32),
        )

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Attention",
                        ["query", "key", "value", "mask", "past_key", "past_value"],
                        ["attn", "present_key", "present_value"],
                        scale=scale,
                        q_num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                    ),
                    # QGA contribops
                    oh.make_node("Shape", ["query"], ["batch"], end=1),
                    oh.make_node("Where", ["mask", "zero", "infty"], ["float_mask"]),
                    oh.make_node("Unsqueeze", ["float_mask", "cst01"], ["expanded_mask"]),
                    oh.make_node("Shape", ["mask"], ["total_seqlength64"], start=-1),
                    oh.make_node(
                        "Cast", ["total_seqlength64"], ["total_seqlength"], to=TensorProto.INT32
                    ),
                    oh.make_node("Sub", ["total_seqlength", "one"], ["total_seqlength_1"]),
                    oh.make_node("Expand", ["total_seqlength_1", "batch"], ["seqlensk"]),
                    oh.make_node("Concat", ["query", "key", "value"], ["packed"], axis=-1),
                    oh.make_node(
                        "GroupQueryAttention",
                        [
                            "packed",
                            "",
                            "",
                            "past_key",
                            "past_value",
                            "seqlensk",
                            "total_seqlength",
                            "",
                            "",
                            "",
                            "expanded_mask",
                        ],
                        ["attn_gqa", "present_key_gqa", "present_value_gqa"],
                        do_rotary=0,
                        num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                        rotary_interleaved=0,
                        scale=scale,
                        domain="com.microsoft",
                    ),
                ],
                "gqa",
                [
                    _mkv_("query", TFLOAT, ["b", "l", "hs"]),
                    _mkv_("key", TFLOAT, ["b", "l2", "h2s"]),
                    _mkv_("value", TFLOAT, ["b", "l2", "h2s"]),
                    _mkv_("mask", TensorProto.BOOL, ["m1", "m2"]),
                    _mkv_("past_key", TFLOAT, ["b", "h3", "lp", "s"]),
                    _mkv_("past_value", TFLOAT, ["b", "h3", "lp", "s"]),
                ],
                [
                    _mkv_("attn", TFLOAT, ["b", "l3", "h3s"]),
                    _mkv_("present_key", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("present_value", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("attn_gqa", TFLOAT, ["b", "l3", "h3s"]),
                    _mkv_("present_key_gqa", TFLOAT, ["b", "ho", "lo", "s"]),
                    _mkv_("present_value_gqa", TFLOAT, ["b", "ho", "lo", "s"]),
                ],
                [
                    # onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="shape00"),
                    # onh.from_array(
                    #    np.array([0, 0, -1, head_size], dtype=np.int64), name="shape0000"
                    # ),
                    onh.from_array(np.array([0, 1], dtype=np.int64), name="cst01"),
                    onh.from_array(np.array([1], dtype=np.int32), name="one"),
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(
                        np.array([np.finfo(np.float32).min], dtype=np.float32), name="infty"
                    ),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 24), oh.make_opsetid("com.microsoft", 1)],
            ir_version=11,
        )
        model = shape_inference.infer_shapes(model)
        check_model(model)
        self.dump_onnx("test_onnx_gqa_no_rotary_3D_packed.onnx", model)

        sess = self._check_with_ort(model, cpu=True)
        feeds = dict(zip([i.name for i in model.graph.input], inputs))
        got = sess.run(None, feeds)
        self.assertEqualArray(got[1], got[4], atol=1e-5)
        self.assertEqualArray(got[2], got[5], atol=1e-5)
        self.assertEqualArray(got[0], got[3], atol=1e-5)

    @ignore_warnings((UserWarning, FutureWarning))
    @hide_stdout()
    def test_multi_head_attention_fused_matmul(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["query"], ["t_query"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["keys"], ["t_keys"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["values"], ["t_values"], perm=[0, 2, 1, 3]),
                    oh.make_node("Concat", ["past_keys", "t_keys"], ["ct_keys"], axis=-2),
                    oh.make_node("Concat", ["past_values", "t_values"], ["ct_values"], axis=-2),
                    oh.make_node("Mul", ["t_query", "scale_sqrt"], ["query_scaled"]),
                    oh.make_node("Mul", ["ct_keys", "scale_sqrt"], ["keys_scaled"]),
                    oh.make_node(
                        "FusedMatMul",
                        ["query_scaled", "keys_scaled"],
                        ["qk"],
                        domain="com.microsoft",
                        transB=1,
                    ),
                    oh.make_node("Where", ["mask", "zero", "minfty"], ["bias"]),
                    oh.make_node("Add", ["qk", "bias"], ["qkb"]),
                    oh.make_node("Softmax", ["qkb"], ["qkbs"], axis=-1),
                    oh.make_node("IsNaN", ["qkbs"], ["nans"]),
                    oh.make_node("Where", ["nans", "zero", "qkbs"], ["filt"]),
                    oh.make_node("MatMul", ["filt", "ct_values"], ["prob"]),
                    oh.make_node("Transpose", ["prob"], ["Y"], perm=[0, 2, 1, 3]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("query", TFLOAT, ["aq", "bq", 8, 64]),
                    oh.make_tensor_value_info("keys", TFLOAT, ["ak", "bk", 8, 64]),
                    oh.make_tensor_value_info("values", TFLOAT, ["av", "bv", 8, 64]),
                    oh.make_tensor_value_info("past_keys", TFLOAT, ["pak", 8, "pck", 64]),
                    oh.make_tensor_value_info("past_values", TFLOAT, ["pav", 8, "pcv", 64]),
                    oh.make_tensor_value_info("mask", TensorProto.BOOL, ["am", 1, "cm", "dm"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["ay", "by", "cy", "dy"])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([-np.inf], dtype=np.float32), name="minfty"),
                    onh.from_array(np.array([0.1**0.5], dtype=np.float32), name="scale_sqrt"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=10,
        )
        feeds = dict(
            query=np.random.randn(32, 128, 8, 64).astype(np.float32),
            keys=np.random.randn(32, 128, 8, 64).astype(np.float32),
            values=np.random.randn(32, 128, 8, 64).astype(np.float32),
            mask=np.random.rand(32, 1, 128, 256) >= 0.5,
            past_keys=np.random.randn(32, 8, 128, 64).astype(np.float32),
            past_values=np.random.randn(32, 8, 128, 64).astype(np.float32),
        )
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        z = ref.run(None, feeds)[0]

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FunctionAttention", "MultiHeadAttention3D"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.dump_onnx("test_multi_head_attention_fused_matmul.onnx", opt_onx)
        self.assertIn("MultiHeadAttention", [n.op_type for n in opt_onx.graph.node])
        ref2 = self.make_inference_session(opt_onx)
        zz = ref2.run(None, feeds)[0]
        self.assertEqualArray(z, zz, atol=1e-3)

    def _build_attention_3d_model(self, same_hidden: bool = True):
        """
        Builds a self-attention model with 3D hidden states and QKV projections.
        This model is designed to be processed by FunctionAttentionPattern, which
        creates LocalAttention_to1 (intermediate domain) with MatMul→Reshape→Transpose
        inputs for Q, K, V. Attention3DPattern can then convert it to Attention (com.microsoft).
        """
        num_heads = 8
        head_size = 32
        hidden_dim = num_heads * head_size  # 256

        Wq = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        Wk = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        Wv = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        scale_val = np.array([head_size**-0.5], dtype=np.float32)
        reshape_shape = np.array([0, 0, num_heads, head_size], dtype=np.int64)
        output_shape = np.array([0, 0, hidden_dim], dtype=np.int64)

        # hidden_q = hidden_k = hidden_v when same_hidden=True (pattern fires),
        # otherwise separate inputs (pattern does NOT fire)
        q_src = "hidden" if same_hidden else "hidden_q"
        k_src = "hidden" if same_hidden else "hidden_k"
        v_src = "hidden" if same_hidden else "hidden_v"

        nodes = [
            # Q projection: MatMul → Mul(scale) → Reshape → Transpose([0,2,1,3])
            oh.make_node("MatMul", [q_src, "Wq"], ["Q_3D"]),
            oh.make_node("Mul", ["Q_3D", "scale"], ["Q_3D_s"]),
            oh.make_node("Reshape", ["Q_3D_s", "reshape_shape"], ["Q_4D"]),
            oh.make_node("Transpose", ["Q_4D"], ["Q_4D_t"], perm=[0, 2, 1, 3]),
            # K projection: MatMul → Mul(scale) → Reshape → Transpose([0,2,3,1])
            oh.make_node("MatMul", [k_src, "Wk"], ["K_3D"]),
            oh.make_node("Mul", ["K_3D", "scale"], ["K_3D_s"]),
            oh.make_node("Reshape", ["K_3D_s", "reshape_shape"], ["K_4D"]),
            oh.make_node("Transpose", ["K_4D"], ["K_4D_t"], perm=[0, 2, 3, 1]),
            # V projection: MatMul → Reshape → Transpose([0,2,1,3])
            oh.make_node("MatMul", [v_src, "Wv"], ["V_3D"]),
            oh.make_node("Reshape", ["V_3D", "reshape_shape"], ["V_4D"]),
            oh.make_node("Transpose", ["V_4D"], ["V_4D_t"], perm=[0, 2, 1, 3]),
            # QK attention scores
            oh.make_node("MatMul", ["Q_4D_t", "K_4D_t"], ["QK"]),
            # Where(mask, QK, -inf) → non-SW variant (inf at index 2)
            oh.make_node("Where", ["mask", "QK", "minfty"], ["masked_QK"]),
            oh.make_node("Softmax", ["masked_QK"], ["attn"], axis=-1),
            oh.make_node("IsNaN", ["attn"], ["isnan"]),
            oh.make_node("Where", ["isnan", "zero", "attn"], ["attn_clean"]),
            oh.make_node("MatMul", ["attn_clean", "V_4D_t"], ["output_4D"]),
            # Output reshape
            oh.make_node("Transpose", ["output_4D"], ["output_4D_t"], perm=[0, 2, 1, 3]),
            oh.make_node("Reshape", ["output_4D_t", "output_shape"], ["output"]),
        ]
        dtype = np.float32
        if same_hidden:
            inputs = [
                oh.make_tensor_value_info("hidden", TFLOAT, ["batch", "seq", hidden_dim]),
                oh.make_tensor_value_info("mask", TensorProto.BOOL, [1, 1, "seq", "seq"]),
            ]
        else:
            inputs = [
                oh.make_tensor_value_info("hidden_q", TFLOAT, ["batch", "seq", hidden_dim]),
                oh.make_tensor_value_info("hidden_k", TFLOAT, ["batch", "seq", hidden_dim]),
                oh.make_tensor_value_info("hidden_v", TFLOAT, ["batch", "seq", hidden_dim]),
                oh.make_tensor_value_info("mask", TensorProto.BOOL, [1, 1, "seq", "seq"]),
            ]
        model = oh.make_model(
            oh.make_graph(
                nodes,
                "attention_3d",
                inputs,
                [oh.make_tensor_value_info("output", TFLOAT, ["batch", "seq", hidden_dim])],
                [
                    onh.from_array(Wq, name="Wq"),
                    onh.from_array(Wk, name="Wk"),
                    onh.from_array(Wv, name="Wv"),
                    onh.from_array(scale_val, name="scale"),
                    onh.from_array(reshape_shape, name="reshape_shape"),
                    onh.from_array(output_shape, name="output_shape"),
                    onh.from_array(np.array([0], dtype=dtype), name="zero"),
                    onh.from_array(np.array([np.finfo(dtype).min], dtype=dtype), name="minfty"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 22)],
            ir_version=10,
        )
        return model, num_heads, head_size, hidden_dim

    def test_attention_3d_pattern(self):
        model, _num_heads, _head_size, hidden_dim = self._build_attention_3d_model(
            same_hidden=True
        )
        # self.dump_onnx("test_attention_3d_pattern_input.onnx", model)
        feeds = dict(
            hidden=self._range(2, 5, hidden_dim),
            mask=np.random.randint(0, 2, size=(1, 1, 5, 5)) % 2 == 1,
        )
        sess = self._check_with_ort(model, cpu=True)
        expected = sess.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FunctionAttention", "Attention3D"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        # self.dump_onnx("test_attention_3d_pattern_output.onnx", opt_onx)
        sess_opt = self._check_with_ort(opt_onx, cpu=False)
        got = sess_opt.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=0.05)

        # After optimization, Attention (com.microsoft) should appear
        op_types = [(n.op_type, n.domain) for n in opt_onx.graph.node]
        self.assertIn(("Attention", "com.microsoft"), op_types)
        # The QKV MatMul/Reshape/Transpose projections should be absorbed
        self.assertNotIn("LocalAttention_to1", (n for n, _ in op_types))

    def test_attention_3d_pattern_no_match_different_hidden(self):
        model, _num_heads, _head_size, _hidden_dim = self._build_attention_3d_model(
            same_hidden=False
        )

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FunctionAttention", "Attention3D"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)

        # Pattern should NOT fire since Q, K, V come from different inputs
        ms_attention = [
            n
            for n in opt_onx.graph.node
            if n.op_type == "Attention" and n.domain == "com.microsoft"
        ]
        self.assertEqual(0, len(ms_attention))
        self.assertIn("LocalAttention_to1", [n.op_type for n in opt_onx.graph.node])

    def _build_relative_position_bias_model(self):
        """Builds a T5-style (bidirectional) relative position bias ONNX model."""
        import math

        num_heads = 8
        num_buckets_total = 32
        max_distance = 128
        half_num_buckets = num_buckets_total // 2  # 16
        max_exact = half_num_buckets // 2  # 8
        log_max = math.log(max_distance / max_exact)  # log(16) ≈ 2.77
        scale = float(half_num_buckets - max_exact)  # 8.0
        clamp_val = half_num_buckets - 1  # 15

        bias_table = (
            np.arange(num_buckets_total * num_heads)
            .reshape(num_buckets_total, num_heads)
            .astype(np.float32)
        )

        nodes = [
            # Range(0, seq_len, 1) -> [seq_len]
            oh.make_node("Range", ["zero", "seq_len", "one"], ["range"]),
            # Unsqueeze(range, [0]) -> [1, seq_len]
            oh.make_node("Unsqueeze", ["range", "axis0"], ["unsqueeze_q"]),
            # Unsqueeze(range, [1]) -> [seq_len, 1]
            oh.make_node("Unsqueeze", ["range", "axis1"], ["unsqueeze_k"]),
            # Sub -> [seq_len, seq_len]
            oh.make_node("Sub", ["unsqueeze_q", "unsqueeze_k"], ["rel_pos"]),
            # Abs
            oh.make_node("Abs", ["rel_pos"], ["abs_pos"]),
            # Cast to int64 (for Where true branch and Less)
            oh.make_node("Cast", ["abs_pos"], ["abs_pos_int"], to=TINT64),
            # Cast to float (for log computation)
            oh.make_node("Cast", ["abs_pos"], ["abs_pos_float"], to=TFLOAT),
            # Less -> condition for Where
            oh.make_node("Less", ["abs_pos_int", "max_exact_int"], ["is_small"]),
            # Div(pos_float, max_exact_float)
            oh.make_node("Div", ["abs_pos_float", "max_exact_float"], ["div_pos"]),
            # Log
            oh.make_node("Log", ["div_pos"], ["log_div"]),
            # Div(log, log_max)
            oh.make_node("Div", ["log_div", "log_max_cst"], ["div_log"]),
            # Mul(div_log, scale)
            oh.make_node("Mul", ["div_log", "scale_cst"], ["mul_result"]),
            # Cast to int64
            oh.make_node("Cast", ["mul_result"], ["bucket_int"], to=TINT64),
            # Add(bucket, max_exact_int)
            oh.make_node("Add", ["bucket_int", "max_exact_int"], ["add_result"]),
            # Shape(add_result)
            oh.make_node("Shape", ["add_result"], ["add_shape"]),
            # ConstantOfShape(shape, clamp_val)
            oh.make_node(
                "ConstantOfShape",
                ["add_shape"],
                ["clamp_cst"],
                value=onh.from_array(np.array([clamp_val], dtype=np.int64)),
            ),
            # Min(add_result, clamp_cst)
            oh.make_node("Min", ["add_result", "clamp_cst"], ["bucket_clamped"]),
            # Where(is_small, abs_pos_int, bucket_clamped)
            oh.make_node(
                "Where", ["is_small", "abs_pos_int", "bucket_clamped"], ["final_bucket"]
            ),
            # Gather(bias_table, final_bucket)
            oh.make_node("Gather", ["bias_table", "final_bucket"], ["gathered"], axis=0),
            # Transpose([2, 0, 1]) -> [num_heads, seq_len, seq_len]
            oh.make_node("Transpose", ["gathered"], ["transposed"], perm=[2, 0, 1]),
            # Unsqueeze([0]) -> [1, num_heads, seq_len, seq_len]
            oh.make_node("Unsqueeze", ["transposed", "batch_axis"], ["rpb_output"]),
        ]

        inits = [
            onh.from_array(np.array(0, dtype=np.int64), name="zero"),
            onh.from_array(np.array(1, dtype=np.int64), name="one"),
            onh.from_array(np.array([0], dtype=np.int64), name="axis0"),
            onh.from_array(np.array([1], dtype=np.int64), name="axis1"),
            onh.from_array(np.array([0], dtype=np.int64), name="batch_axis"),
            onh.from_array(np.array(max_exact, dtype=np.int64), name="max_exact_int"),
            onh.from_array(np.array(float(max_exact), dtype=np.float32), name="max_exact_float"),
            onh.from_array(np.array(log_max, dtype=np.float32), name="log_max_cst"),
            onh.from_array(np.array(scale, dtype=np.float32), name="scale_cst"),
            onh.from_array(bias_table, name="bias_table"),
        ]

        model = oh.make_model(
            oh.make_graph(
                nodes,
                "rpb_test",
                [oh.make_tensor_value_info("seq_len", TINT64, [])],
                [oh.make_tensor_value_info("rpb_output", TFLOAT, [1, num_heads, None, None])],
                inits,
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_relative_position_bias(self):
        """Tests that RelativePositionBiasPattern fuses the T5 encoder subgraph."""
        model = self._build_relative_position_bias_model()
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["RelativePositionBias"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)

        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("RelativePositionBias", op_types)
        self.assertNotIn("Gather", op_types)
        self.assertNotIn("Where", op_types)

        rpb_nodes = [n for n in opt_onx.graph.node if n.op_type == "RelativePositionBias"]
        self.assertEqual(1, len(rpb_nodes))
        rpb_node = rpb_nodes[0]
        self.assertEqual("com.microsoft", rpb_node.domain)

        attr_map = {a.name: a for a in rpb_node.attribute}
        self.assertIn("max_distance", attr_map)
        self.assertEqual(128, attr_map["max_distance"].i)
        self.assertIn("is_bidirectional", attr_map)
        self.assertEqual(1, attr_map["is_bidirectional"].i)

        # Bias table should now be transposed: [num_heads, num_buckets]
        init_names = {init.name for init in opt_onx.graph.initializer}
        self.assertIn(rpb_node.input[0], init_names)

    def test_relative_position_bias_in_pattern_list(self):
        """Tests that RelativePositionBiasPattern is in the default ORT pattern list."""
        from yobx.xoptim.patterns_ort import get_onnxruntime_patterns

        patterns = get_onnxruntime_patterns()
        names = [p.__class__.__name__ for p in patterns]
        self.assertIn("RelativePositionBiasPattern", names)

    def _build_gated_relative_position_bias_model(self):
        """Constructs a DeBERTa-style gated relative position bias ONNX model."""
        batch_size = 2
        seq_len_val = 5
        num_heads = 4
        head_size = 8
        D = 4  # gate_ur_linear output dim, must be even; D//2 = 2

        # Constants for shapes
        # query_layer: (batch, seq_len, num_heads * head_size) = (2, 5, 32)
        query_hidden = num_heads * head_size  # 32

        query_bias = np.zeros(query_hidden, dtype=np.float32)
        gate_weight = np.arange(head_size * D, dtype=np.float32).reshape(head_size, D) * 0.01
        gate_bias = np.zeros(D, dtype=np.float32)
        eco_a = np.ones((1, num_heads, 1, 1), dtype=np.float32)

        # Reshape targets
        reshape1_shape = np.array([0, 0, num_heads, head_size], dtype=np.int64)
        reshape2_shape = np.array([0, num_heads, 0, 2, D // 2], dtype=np.int64)

        nodes = [
            # Add(query_layer, query_bias)
            oh.make_node("Add", ["query_layer", "query_bias"], ["added_query"]),
            # Reshape to (batch, seq_len, num_heads, head_size)
            oh.make_node("Reshape", ["added_query", "reshape1_shape"], ["reshaped_q"]),
            # Transpose to (batch, num_heads, seq_len, head_size)
            oh.make_node("Transpose", ["reshaped_q"], ["transposed_q"], perm=[0, 2, 1, 3]),
            # MatMul with gate_weight -> (batch, num_heads, seq_len, D)
            oh.make_node("MatMul", ["transposed_q", "gate_weight"], ["gate_mm"]),
            # Add gate_bias
            oh.make_node("Add", ["gate_mm", "gate_bias"], ["gate_biased"]),
            # Reshape to (batch, num_heads, seq_len, 2, D//2)
            oh.make_node("Reshape", ["gate_biased", "reshape2_shape"], ["gate_r2"]),
            # ReduceSum along axis=-1, keepdims=0
            oh.make_node("ReduceSum", ["gate_r2", "reduce_axis"], ["gate_sum"], keepdims=0),
            # Sigmoid
            oh.make_node("Sigmoid", ["gate_sum"], ["gate_sig"]),
            # Split along axis=-1 -> gate_u (output[0]), gate_r (output[1])
            oh.make_node("Split", ["gate_sig", "split_sizes"], ["gate_u", "gate_r"], axis=-1),
            # Mul(gate_r, eco_a)
            oh.make_node("Mul", ["gate_r", "eco_a"], ["gate_r_eco"]),
            # Sub(gate_r_eco, 1.0)
            oh.make_node("Sub", ["gate_r_eco", "one_f"], ["gate_r_eco_sub"]),
            # Mul(gate_u, gate_r_eco_sub)
            oh.make_node("Mul", ["gate_u", "gate_r_eco_sub"], ["gate_u_mul"]),
            # Add(gate_u_mul, 2.0)
            oh.make_node("Add", ["gate_u_mul", "two_f"], ["gate_u_1"]),
            # Mul(gate_u_1, rel_pos)
            oh.make_node("Mul", ["gate_u_1", "rel_pos"], ["output"]),
        ]

        inits = [
            onh.from_array(query_bias, name="query_bias"),
            onh.from_array(gate_weight, name="gate_weight"),
            onh.from_array(gate_bias, name="gate_bias"),
            onh.from_array(eco_a, name="eco_a"),
            onh.from_array(reshape1_shape, name="reshape1_shape"),
            onh.from_array(reshape2_shape, name="reshape2_shape"),
            onh.from_array(np.array([-1], dtype=np.int64), name="reduce_axis"),
            onh.from_array(np.array([1, 1], dtype=np.int64), name="split_sizes"),
            onh.from_array(np.array(1.0, dtype=np.float32), name="one_f"),
            onh.from_array(np.array(2.0, dtype=np.float32), name="two_f"),
        ]

        model = oh.make_model(
            oh.make_graph(
                nodes,
                "grpb_test",
                [
                    oh.make_tensor_value_info(
                        "query_layer", TFLOAT, [batch_size, seq_len_val, query_hidden]
                    ),
                    oh.make_tensor_value_info(
                        "rel_pos", TFLOAT, [1, num_heads, seq_len_val, seq_len_val]
                    ),
                ],
                [
                    oh.make_tensor_value_info(
                        "output", TFLOAT, [batch_size, num_heads, seq_len_val, seq_len_val]
                    )
                ],
                inits,
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_gated_relative_position_bias(self):
        """Verifies that GatedRelativePositionBiasPattern fuses the DeBERTa gating subgraph."""
        model = self._build_gated_relative_position_bias_model()
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["GatedRelativePositionBias"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)

        op_types = [(n.op_type, n.domain) for n in opt_onx.graph.node]
        self.assertIn(("GatedRelativePositionBias", "com.microsoft"), op_types)
        self.assertNotIn(("Split", ""), op_types)
        self.assertNotIn(("Sigmoid", ""), op_types)
        self.assertNotIn(("ReduceSum", ""), op_types)

        grpb_nodes = [
            n
            for n in opt_onx.graph.node
            if n.op_type == "GatedRelativePositionBias" and n.domain == "com.microsoft"
        ]
        self.assertEqual(1, len(grpb_nodes))

        attr_map = {a.name: a for a in grpb_nodes[0].attribute}
        self.assertIn("num_heads", attr_map)
        self.assertEqual(4, attr_map["num_heads"].i)

    def test_gated_relative_position_bias_in_pattern_list(self):
        """Verifies that GatedRelativePositionBiasPattern is in the default ORT pattern list."""
        from yobx.xoptim.patterns_ort import get_onnxruntime_patterns

        patterns = get_onnxruntime_patterns()
        names = [p.__class__.__name__ for p in patterns]
        self.assertIn("GatedRelativePositionBiasPattern", names)

    """Tests for MissingReduceMaxPattern and MissingTopKPattern."""

    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_missing_reduce_max_pattern_matches_bfloat16(self):
        """MissingReduceMaxPattern wraps ReduceMax on BFLOAT16 with Cast nodes."""
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["X"], ["Xbf"], to=TensorProto.BFLOAT16),
                    oh.make_node("ReduceMax", ["Xbf"], ["Y"], keepdims=0),
                    oh.make_node("Cast", ["Y"], ["Yf"], to=TFLOAT),
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, [3, 4])],
                [oh.make_tensor_value_info("Yf", TFLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=["MissingReduceMax"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        # The ReduceMax on BFLOAT16 should be surrounded by Cast nodes
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("ReduceMax", op_types)
        # There should be extra Cast nodes introduced by the pattern
        cast_count = op_types.count("Cast")
        self.assertGreaterEqual(cast_count, 3)

    def test_missing_reduce_max_pattern_no_match_float(self):
        """MissingReduceMaxPattern does not fire for FLOAT input."""
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("ReduceMax", ["X"], ["Y"], keepdims=0)],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, [3, 4])],
                [oh.make_tensor_value_info("Y", TFLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=["MissingReduceMax"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        # No Cast should have been introduced
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertNotIn("Cast", op_types)

    def test_missing_topk_pattern_matches_bfloat16(self):
        """MissingTopKPattern wraps TopK on BFLOAT16 with Cast nodes."""
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["X"], ["Xbf"], to=TensorProto.BFLOAT16),
                    oh.make_node("TopK", ["Xbf", "k"], ["vals", "inds"]),
                    oh.make_node("Cast", ["vals"], ["valsf"], to=TFLOAT),
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, [3, 4])],
                [
                    oh.make_tensor_value_info("valsf", TFLOAT, [3, 2]),
                    oh.make_tensor_value_info("inds", TINT64, [3, 2]),
                ],
                [onh.from_array(np.array([2], dtype=np.int64), name="k")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=["MissingTopK"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("TopK", op_types)
        # Extra Cast nodes should be present
        cast_count = op_types.count("Cast")
        self.assertGreaterEqual(cast_count, 3)

    def test_missing_topk_pattern_no_match_float(self):
        """MissingTopKPattern does not fire for FLOAT input."""
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("TopK", ["X", "k"], ["vals", "inds"])],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, [3, 4])],
                [
                    oh.make_tensor_value_info("vals", TFLOAT, [3, 2]),
                    oh.make_tensor_value_info("inds", TINT64, [3, 2]),
                ],
                [onh.from_array(np.array([2], dtype=np.int64), name="k")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=["MissingTopK"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertNotIn("Cast", op_types)


class TestComplexMulPatterns(ExtTestCase):
    """Tests for ComplexMulPattern and ComplexMulConjPattern."""

    def _make_complex_mul_model(self, shape, conj: bool = False):
        """Builds a model that computes complex multiplication of A and B.

        When ``conj`` is ``True``, the imaginary combination uses Sub instead
        of Add (corresponding to ``ComplexMulConj``).
        """
        TFLOAT_ = TensorProto.FLOAT
        cst0 = onh.from_array(np.array(0, dtype=np.int64), name="cst0")
        cst1 = onh.from_array(np.array(1, dtype=np.int64), name="cst1")
        neg1 = onh.from_array(np.array([-1], dtype=np.int64), name="neg1")

        #   A_r = Gather(A, 0, axis=-1)
        #   A_i = Gather(A, 1, axis=-1)
        #   B_r = Gather(B, 0, axis=-1)
        #   B_i = Gather(B, 1, axis=-1)
        #   t_rr = A_r * B_r
        #   t_ii = A_i * B_i
        #   t_ri = A_r * B_i
        #   t_ir = A_i * B_r
        #   C_r = Sub(t_rr, t_ii)
        #   C_i = Add(t_ri, t_ir)  -- or Sub(t_ir, t_ri) for conj
        #   C = Concat([Unsqueeze(C_r, -1), Unsqueeze(C_i, -1)], axis=-1)

        if not conj:
            real_op = oh.make_node("Sub", ["t_rr", "t_ii"], ["c_r"])
            imag_op = oh.make_node("Add", ["t_ri", "t_ir"], ["c_i"])
        else:
            real_op = oh.make_node("Add", ["t_rr", "t_ii"], ["c_r"])
            imag_op = oh.make_node("Sub", ["t_ir", "t_ri"], ["c_i"])

        nodes = [
            oh.make_node("Gather", ["A", "cst0"], ["a_r"], axis=-1),
            oh.make_node("Gather", ["A", "cst1"], ["a_i"], axis=-1),
            oh.make_node("Gather", ["B", "cst0"], ["b_r"], axis=-1),
            oh.make_node("Gather", ["B", "cst1"], ["b_i"], axis=-1),
            oh.make_node("Mul", ["a_r", "b_r"], ["t_rr"]),
            oh.make_node("Mul", ["a_i", "b_i"], ["t_ii"]),
            oh.make_node("Mul", ["a_r", "b_i"], ["t_ri"]),
            oh.make_node("Mul", ["a_i", "b_r"], ["t_ir"]),
            real_op,
            imag_op,
            oh.make_node("Unsqueeze", ["c_r", "neg1"], ["c_r_u"]),
            oh.make_node("Unsqueeze", ["c_i", "neg1"], ["c_i_u"]),
            oh.make_node("Concat", ["c_r_u", "c_i_u"], ["C"], axis=-1),
        ]
        model = oh.make_model(
            oh.make_graph(
                nodes,
                "complex_mul",
                [
                    oh.make_tensor_value_info("A", TFLOAT_, shape),
                    oh.make_tensor_value_info("B", TFLOAT_, shape),
                ],
                [oh.make_tensor_value_info("C", TFLOAT_, shape)],
                [cst0, cst1, neg1],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        return model

    def test_complex_mul_pattern_match(self):
        """ComplexMulPattern fuses the decomposed complex multiplication."""
        shape = [2, 4, 2]
        model = self._make_complex_mul_model(shape, conj=False)

        rng = np.random.default_rng(0)
        feeds = {
            "A": rng.standard_normal(shape).astype(np.float32),
            "B": rng.standard_normal(shape).astype(np.float32),
        }

        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["ComplexMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("ComplexMul", op_types)
        fused = opt_onx.graph.node[0]
        self.assertEqual(fused.op_type, "ComplexMul")
        self.assertEqual(fused.domain, "com.microsoft")

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_complex_mul_conj_pattern_match(self):
        """ComplexMulConjPattern fuses the decomposed complex multiplication with conjugate."""
        shape = [2, 4, 2]
        model = self._make_complex_mul_model(shape, conj=True)

        rng = np.random.default_rng(1)
        feeds = {
            "A": rng.standard_normal(shape).astype(np.float32),
            "B": rng.standard_normal(shape).astype(np.float32),
        }

        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["ComplexMulConj"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("ComplexMulConj", op_types)
        fused = opt_onx.graph.node[0]
        self.assertEqual(fused.op_type, "ComplexMulConj")
        self.assertEqual(fused.domain, "com.microsoft")

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_complex_mul_pattern_no_match_shared_mul(self):
        """ComplexMulPattern must not fire when a Mul output is consumed elsewhere."""
        shape = [2, 4, 2]
        # Build model where t_rr is also used as an extra output.
        cst0 = onh.from_array(np.array(0, dtype=np.int64), name="cst0")
        cst1 = onh.from_array(np.array(1, dtype=np.int64), name="cst1")
        neg1 = onh.from_array(np.array([-1], dtype=np.int64), name="neg1")
        TFLOAT_ = TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Gather", ["A", "cst0"], ["a_r"], axis=-1),
                    oh.make_node("Gather", ["A", "cst1"], ["a_i"], axis=-1),
                    oh.make_node("Gather", ["B", "cst0"], ["b_r"], axis=-1),
                    oh.make_node("Gather", ["B", "cst1"], ["b_i"], axis=-1),
                    oh.make_node("Mul", ["a_r", "b_r"], ["t_rr"]),
                    oh.make_node("Mul", ["a_i", "b_i"], ["t_ii"]),
                    oh.make_node("Mul", ["a_r", "b_i"], ["t_ri"]),
                    oh.make_node("Mul", ["a_i", "b_r"], ["t_ir"]),
                    oh.make_node("Sub", ["t_rr", "t_ii"], ["c_r"]),
                    oh.make_node("Add", ["t_ri", "t_ir"], ["c_i"]),
                    oh.make_node("Unsqueeze", ["c_r", "neg1"], ["c_r_u"]),
                    oh.make_node("Unsqueeze", ["c_i", "neg1"], ["c_i_u"]),
                    oh.make_node("Concat", ["c_r_u", "c_i_u"], ["C"], axis=-1),
                    # t_rr is used again here, making the pattern unsafe to remove.
                    oh.make_node("Relu", ["t_rr"], ["extra"]),
                ],
                "complex_mul_shared",
                [
                    oh.make_tensor_value_info("A", TFLOAT_, shape),
                    oh.make_tensor_value_info("B", TFLOAT_, shape),
                ],
                [
                    oh.make_tensor_value_info("C", TFLOAT_, shape),
                    oh.make_tensor_value_info("extra", TFLOAT_, shape[:-1]),
                ],
                [cst0, cst1, neg1],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        check_model(model)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["ComplexMul"]),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        # The pattern should NOT have been applied.
        self.assertNotIn("ComplexMul", op_types)


class TestCausalConvWithStatePattern(ExtTestCase):
    """Tests for CausalConvWithStatePattern."""

    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def _make_causal_conv_model(
        self,
        batch: int = 1,
        channels: int = 4,
        seq_len: int = 8,
        kernel_size: int = 3,
        with_bias: bool = True,
        with_slice: bool = True,
    ) -> "ModelProto":
        """Builds a Concat + Conv (+ Slice) graph for testing."""
        state_len = kernel_size - 1
        concat_output_len = state_len + seq_len

        nodes = [
            oh.make_node("Concat", ["past_state", "input"], ["concat_out"], axis=2),
            oh.make_node(
                "Conv",
                ["concat_out", "weight"] + (["bias"] if with_bias else []),
                ["output"],
                group=channels,
                pads=[0, 0],
            ),
        ]
        if with_slice:
            nodes.append(
                oh.make_node(
                    "Slice",
                    ["concat_out", "slice_starts", "slice_ends", "slice_axes"],
                    ["present_state"],
                )
            )

        inputs = [
            oh.make_tensor_value_info("input", TFLOAT, [batch, channels, seq_len]),
            oh.make_tensor_value_info("past_state", TFLOAT, [batch, channels, state_len]),
        ]
        graph_outputs = [oh.make_tensor_value_info("output", TFLOAT, [batch, channels, seq_len])]
        if with_slice:
            graph_outputs.append(
                oh.make_tensor_value_info("present_state", TFLOAT, [batch, channels, state_len])
            )

        initializers = [
            onh.from_array(
                np.random.randn(channels, 1, kernel_size).astype(np.float32), name="weight"
            )
        ]
        if with_bias:
            initializers.append(onh.from_array(np.zeros(channels, dtype=np.float32), name="bias"))
        if with_slice:
            initializers += [
                onh.from_array(np.array([seq_len], dtype=np.int64), name="slice_starts"),
                onh.from_array(np.array([concat_output_len], dtype=np.int64), name="slice_ends"),
                onh.from_array(np.array([2], dtype=np.int64), name="slice_axes"),
            ]

        return oh.make_model(
            oh.make_graph(nodes, "causal_conv", inputs, graph_outputs, initializers),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

    def test_causal_conv_with_state_pattern_in_list(self):
        """CausalConvWithStatePattern must appear in the default ORT pattern list."""
        from yobx.xoptim.patterns_ort import get_onnxruntime_patterns

        patterns = get_onnxruntime_patterns()
        names = [p.__class__.__name__ for p in patterns]
        self.assertIn("CausalConvWithStatePattern", names)

    def test_causal_conv_with_state_basic(self):
        """Concat + depthwise Conv + Slice fuses to CausalConvWithState."""
        model = self._make_causal_conv_model(with_slice=True)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["CausalConvWithState"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("CausalConvWithState", op_types)
        self.assertNotIn("Concat", op_types)
        self.assertNotIn("Slice", op_types)
        # The fused node must live in the com.microsoft domain.
        fused_nodes = [n for n in opt_onx.graph.node if n.op_type == "CausalConvWithState"]
        self.assertEqual(1, len(fused_nodes))
        self.assertEqual("com.microsoft", fused_nodes[0].domain)
        # Two outputs: convolution result + present_state.
        self.assertEqual(2, len(fused_nodes[0].output))

    def test_causal_conv_with_state_no_slice(self):
        """Concat + depthwise Conv (no Slice) also fuses to CausalConvWithState."""
        model = self._make_causal_conv_model(with_slice=False)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["CausalConvWithState"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("CausalConvWithState", op_types)
        self.assertNotIn("Concat", op_types)
        fused_nodes = [n for n in opt_onx.graph.node if n.op_type == "CausalConvWithState"]
        self.assertEqual(1, len(fused_nodes))
        self.assertEqual("com.microsoft", fused_nodes[0].domain)
        # Single output: only the convolution result.
        self.assertEqual(1, len(fused_nodes[0].output))

    def test_causal_conv_with_state_no_bias(self):
        """Fusion also works when the Conv has no bias."""
        model = self._make_causal_conv_model(with_bias=False, with_slice=True)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["CausalConvWithState"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("CausalConvWithState", op_types)
        self.assertNotIn("Concat", op_types)

    def test_causal_conv_no_match_not_depthwise(self):
        """Pattern must NOT fire when groups < channels (not depthwise)."""
        # groups=1, channels=4 → not a depthwise convolution
        batch, channels, seq_len, kernel_size = 1, 4, 8, 3
        state_len = kernel_size - 1
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["past_state", "input"], ["concat_out"], axis=2),
                    oh.make_node(
                        "Conv", ["concat_out", "weight"], ["output"], group=1, pads=[0, 0]
                    ),
                ],
                "no_depthwise",
                [
                    oh.make_tensor_value_info("input", TFLOAT, [batch, channels, seq_len]),
                    oh.make_tensor_value_info("past_state", TFLOAT, [batch, channels, state_len]),
                ],
                [oh.make_tensor_value_info("output", TFLOAT, [batch, channels, seq_len])],
                [
                    onh.from_array(
                        np.random.randn(channels, channels, kernel_size).astype(np.float32),
                        name="weight",
                    )
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["CausalConvWithState"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertNotIn("CausalConvWithState", op_types)

    def test_causal_conv_no_match_with_padding(self):
        """Pattern must NOT fire when the Conv has non-zero pads."""
        batch, channels, seq_len, kernel_size = 1, 4, 8, 3
        state_len = kernel_size - 1
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["past_state", "input"], ["concat_out"], axis=2),
                    oh.make_node(
                        "Conv",
                        ["concat_out", "weight"],
                        ["output"],
                        group=channels,
                        pads=[1, 1],  # non-zero padding
                    ),
                ],
                "padded_conv",
                [
                    oh.make_tensor_value_info("input", TFLOAT, [batch, channels, seq_len]),
                    oh.make_tensor_value_info("past_state", TFLOAT, [batch, channels, state_len]),
                ],
                [oh.make_tensor_value_info("output", TFLOAT, [batch, channels, seq_len])],
                [
                    onh.from_array(
                        np.random.randn(channels, 1, kernel_size).astype(np.float32),
                        name="weight",
                    )
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["CausalConvWithState"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertNotIn("CausalConvWithState", op_types)

    def test_fused_matmul_activation_relu(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("FusedMatMul", ["X", "Y"], ["mm"], domain="com.microsoft"),
                    oh.make_node("Relu", ["mm"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 2, 32, 64]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 2, 64, 16]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 2, 32, 16])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        feeds = {"X": self._range(2, 2, 32, 64), "Y": self._range(2, 2, 64, 16)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FusedMatMulActivation"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["FusedMatMulActivation"], [n.op_type for n in opt_onx.graph.node])
        node = opt_onx.graph.node[0]
        self.assertEqual(node.domain, "com.microsoft")
        act_attr = {a.name: a for a in node.attribute}
        self.assertEqual(act_attr["activation"].s.decode(), "Relu")

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])

    def test_fused_matmul_activation_from_matmul(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MatMul", ["X", "Y"], ["mm"]), oh.make_node("Tanh", ["mm"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [4, 8]),
                    oh.make_tensor_value_info("Y", TFLOAT, [8, 16]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [4, 16])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        feeds = {"X": self._range(4, 8), "Y": self._range(8, 16)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FusedMatMulActivation"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["FusedMatMulActivation"], [n.op_type for n in opt_onx.graph.node])
        node = opt_onx.graph.node[0]
        self.assertEqual(node.domain, "com.microsoft")
        act_attr = {a.name: a for a in node.attribute}
        self.assertEqual(act_attr["activation"].s.decode(), "Tanh")

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-6)

    def test_fused_matmul_activation_leaky_relu(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("FusedMatMul", ["X", "Y"], ["mm"], domain="com.microsoft"),
                    oh.make_node("LeakyRelu", ["mm"], ["Z"], alpha=0.1),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [4, 8]),
                    oh.make_tensor_value_info("Y", TFLOAT, [8, 16]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [4, 16])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        feeds = {"X": self._range(4, 8, bias=-0.5), "Y": self._range(8, 16)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FusedMatMulActivation"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(["FusedMatMulActivation"], [n.op_type for n in opt_onx.graph.node])
        node = opt_onx.graph.node[0]
        self.assertEqual(node.domain, "com.microsoft")
        act_attr = {a.name: a for a in node.attribute}
        self.assertEqual(act_attr["activation"].s.decode(), "LeakyRelu")
        self.assertAlmostEqual(act_attr["activation_alpha"].f, 0.1, atol=1e-5)

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-6)

    def test_fused_matmul_activation_no_fusion_when_used_twice(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("FusedMatMul", ["X", "Y"], ["mm"], domain="com.microsoft"),
                    oh.make_node("Relu", ["mm"], ["r"]),
                    oh.make_node("Add", ["mm", "r"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [4, 8]),
                    oh.make_tensor_value_info("Y", TFLOAT, [8, 16]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [4, 16])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FusedMatMulActivation"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        # The matmul output is used twice so fusion should NOT happen.
        self.assertNotIn("FusedMatMulActivation", [n.op_type for n in opt_onx.graph.node])

    # ---------------------------------------------------------------
    # Tests for GreedySearchPattern
    # ---------------------------------------------------------------

    def _build_greedy_search_model(self, input_dtype=TensorProto.INT64):
        """Builds a minimal ONNX model containing a com.microsoft.GreedySearch node.

        The decoder subgraph is a trivial Cast(INT32→FLOAT) that satisfies the
        type constraints without executing any real language-model logic.
        """
        from onnx import AttributeProto

        decoder_graph = oh.make_graph(
            [oh.make_node("Cast", ["input_ids"], ["logits"], to=TensorProto.FLOAT)],
            "decoder",
            [oh.make_tensor_value_info("input_ids", TensorProto.INT32, [1, None])],
            [oh.make_tensor_value_info("logits", TensorProto.FLOAT, [1, None])],
        )

        gs_node = oh.make_node(
            "GreedySearch",
            ["input_ids", "max_length"],
            ["sequences"],
            domain="com.microsoft",
            eos_token_id=1,
            pad_token_id=0,
        )
        decoder_attr = AttributeProto()
        decoder_attr.name = "decoder"
        decoder_attr.g.CopyFrom(decoder_graph)
        decoder_attr.type = AttributeProto.GRAPH
        gs_node.attribute.append(decoder_attr)

        model = oh.make_model(
            oh.make_graph(
                [gs_node],
                "greedy_search_test",
                [
                    oh.make_tensor_value_info("input_ids", input_dtype, [2, 5]),
                    oh.make_tensor_value_info("max_length", input_dtype, [1]),
                ],
                [oh.make_tensor_value_info("sequences", TensorProto.INT32, [2, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        return model

    def test_greedy_search_int64_cast(self):
        """GreedySearchPattern inserts Cast(INT64→INT32) for integer inputs."""
        model = self._build_greedy_search_model(input_dtype=TINT64)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=["GreedySearch"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)

        op_types = [(n.op_type, n.domain) for n in opt_onx.graph.node]
        self.assertIn(("Cast", ""), op_types)
        self.assertIn(("GreedySearch", "com.microsoft"), op_types)

        gs_nodes = [
            n
            for n in opt_onx.graph.node
            if n.op_type == "GreedySearch" and n.domain == "com.microsoft"
        ]
        self.assertEqual(1, len(gs_nodes))

    def test_greedy_search_no_cast_when_int32(self):
        """GreedySearchPattern does not fire when inputs are already INT32."""
        model = self._build_greedy_search_model(input_dtype=TensorProto.INT32)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=["GreedySearch"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)

        op_types = [(n.op_type, n.domain) for n in opt_onx.graph.node]
        # No Cast nodes should be introduced
        self.assertNotIn(("Cast", ""), op_types)
        self.assertIn(("GreedySearch", "com.microsoft"), op_types)

    def test_greedy_search_in_pattern_list(self):
        """GreedySearchPattern is in the default ORT pattern list."""
        from yobx.xoptim.patterns_ort import get_onnxruntime_patterns

        patterns = get_onnxruntime_patterns()
        names = [p.__class__.__name__ for p in patterns]
        self.assertIn("GreedySearchPattern", names)

    def test_greedy_search_shape_inference(self):
        """Shape inference sets output type to INT32 with rank 2 for GreedySearch."""
        model = self._build_greedy_search_model(input_dtype=TensorProto.INT32)
        gr = GraphBuilder(
            model,
            infer_shapes_options=InferShapesOptions.BUILDER,
            optimization_options=OptimizationOptions(patterns=[]),
        )
        opt_onx = gr.to_onnx(optimize=False)

        output_types = {vi.name: vi.type.tensor_type.elem_type for vi in opt_onx.graph.output}
        self.assertEqual(output_types.get("sequences"), TensorProto.INT32)

    def test_embed_layer_normalization_3_embeddings(self):
        from onnxruntime import InferenceSession

        vocab_size, pos_size, seg_size, hidden = 100, 20, 2, 16
        np.random.seed(42)
        word_table = np.random.randn(vocab_size, hidden).astype(np.float32) * 0.1
        pos_table = np.random.randn(pos_size, hidden).astype(np.float32) * 0.1
        seg_table = np.random.randn(seg_size, hidden).astype(np.float32) * 0.1
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Gather", ["word_table", "input_ids"], ["word_embed"]),
                    oh.make_node("Gather", ["pos_table", "position_ids"], ["pos_embed"]),
                    oh.make_node("Gather", ["seg_table", "segment_ids"], ["seg_embed"]),
                    oh.make_node("Add", ["word_embed", "pos_embed"], ["add1"]),
                    oh.make_node("Add", ["add1", "seg_embed"], ["add2"]),
                    oh.make_node(
                        "LayerNormalization",
                        ["add2", "gamma", "beta"],
                        ["output"],
                        axis=-1,
                        epsilon=1e-5,
                    ),
                ],
                "test",
                [
                    oh.make_tensor_value_info("input_ids", TINT64, ["B", "S"]),
                    oh.make_tensor_value_info("segment_ids", TINT64, ["B", "S"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["B", "S"]),
                ],
                [oh.make_tensor_value_info("output", TFLOAT, ["B", "S", hidden])],
                [
                    onh.from_array(word_table, name="word_table"),
                    onh.from_array(pos_table, name="pos_table"),
                    onh.from_array(seg_table, name="seg_table"),
                    onh.from_array(gamma, name="gamma"),
                    onh.from_array(beta, name="beta"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        B, S = 2, 5
        feeds = {
            "input_ids": np.random.randint(0, vocab_size, (B, S)).astype(np.int64),
            "segment_ids": np.random.randint(0, seg_size, (B, S)).astype(np.int64),
            "position_ids": np.tile(np.arange(S, dtype=np.int64), (B, 1)),
        }
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["EmbedLayerNormalization"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("EmbedLayerNormalization", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_embed_layer_normalization_2_embeddings(self):
        from onnxruntime import InferenceSession

        vocab_size, pos_size, hidden = 100, 20, 16
        np.random.seed(42)
        word_table = np.random.randn(vocab_size, hidden).astype(np.float32) * 0.1
        pos_table = np.random.randn(pos_size, hidden).astype(np.float32) * 0.1
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Gather", ["word_table", "input_ids"], ["word_embed"]),
                    oh.make_node("Gather", ["pos_table", "position_ids"], ["pos_embed"]),
                    oh.make_node("Add", ["word_embed", "pos_embed"], ["add1"]),
                    oh.make_node(
                        "LayerNormalization",
                        ["add1", "gamma", "beta"],
                        ["output"],
                        axis=-1,
                        epsilon=1e-5,
                    ),
                ],
                "test",
                [
                    oh.make_tensor_value_info("input_ids", TINT64, ["B", "S"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["B", "S"]),
                ],
                [oh.make_tensor_value_info("output", TFLOAT, ["B", "S", hidden])],
                [
                    onh.from_array(word_table, name="word_table"),
                    onh.from_array(pos_table, name="pos_table"),
                    onh.from_array(gamma, name="gamma"),
                    onh.from_array(beta, name="beta"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        B, S = 2, 5
        feeds = {
            "input_ids": np.random.randint(0, vocab_size, (B, S)).astype(np.int64),
            "position_ids": np.tile(np.arange(S, dtype=np.int64), (B, 1)),
        }
        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["EmbedLayerNormalization"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("EmbedLayerNormalization", [n.op_type for n in opt_onx.graph.node])
        self.assertIn("com.microsoft", [n.domain for n in opt_onx.graph.node])

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_embed_layer_normalization_no_fusion_without_bias(self):
        vocab_size, pos_size, hidden = 100, 20, 16
        np.random.seed(42)
        word_table = np.random.randn(vocab_size, hidden).astype(np.float32) * 0.1
        pos_table = np.random.randn(pos_size, hidden).astype(np.float32) * 0.1
        gamma = np.ones(hidden, dtype=np.float32)

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Gather", ["word_table", "input_ids"], ["word_embed"]),
                    oh.make_node("Gather", ["pos_table", "position_ids"], ["pos_embed"]),
                    oh.make_node("Add", ["word_embed", "pos_embed"], ["add1"]),
                    # No bias (beta) provided - should NOT fuse
                    oh.make_node(
                        "LayerNormalization", ["add1", "gamma"], ["output"], axis=-1, epsilon=1e-5
                    ),
                ],
                "test",
                [
                    oh.make_tensor_value_info("input_ids", TINT64, ["B", "S"]),
                    oh.make_tensor_value_info("position_ids", TINT64, ["B", "S"]),
                ],
                [oh.make_tensor_value_info("output", TFLOAT, ["B", "S", hidden])],
                [
                    onh.from_array(word_table, name="word_table"),
                    onh.from_array(pos_table, name="pos_table"),
                    onh.from_array(gamma, name="gamma"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["EmbedLayerNormalization"], verbose=0
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        # EmbedLayerNormalization requires both gamma and beta - should NOT fuse
        self.assertNotIn("EmbedLayerNormalization", [n.op_type for n in opt_onx.graph.node])

    # -----------------------------------------------------------------------
    # MoE tests
    # -----------------------------------------------------------------------

    def _make_moe_model(
        self,
        num_tokens: int = 4,
        hidden_size: int = 8,
        inter_size: int = 16,
        num_experts: int = 4,
        top_k: int = 1,
        with_fc1_bias: bool = True,
        with_fc2_bias: bool = True,
        activation: str = "Relu",
    ) -> "ModelProto":
        """Builds the pre-fusion MoE ONNX subgraph for testing.

        The subgraph implements:
            1. Softmax(router_logits, axis=-1) -> router_probs  shape (T, E)
            2. TopK(router_probs, k) -> (top_weights, top_ids)   shapes (T, k)
            3. flat_ids = Reshape(top_ids, (T*k,))
            4. Gather expert FC1/FC2 weights and biases by flat_ids
            5. BatchMatMul for FC1, optional bias, activation
            6. BatchMatMul for FC2, optional bias
            7. Weighted sum using Reshape(top_weights, (T*k, 1))
        Graph input ``router_logits`` contains raw (pre-softmax) routing scores.
        Input ``input`` has shape ``(T*k, H)`` — the pre-dispatched expert tokens.
        """
        T, H, inter, E, k = num_tokens, hidden_size, inter_size, num_experts, top_k

        nodes = []
        initializers = []

        # ----- Softmax over routing logits --------------------------------
        nodes.append(oh.make_node("Softmax", ["router_logits"], ["router_probs"], axis=-1))

        # ----- TopK -------------------------------------------------------
        topk_k = onh.from_array(np.array([k], dtype=np.int64), name="topk_k")
        initializers.append(topk_k)
        nodes.append(
            oh.make_node("TopK", ["router_probs", "topk_k"], ["top_weights", "top_ids"], axis=1)
        )

        # ----- Reshape indices to (T*k,) ----------------------------------
        flat_shape = onh.from_array(np.array([T * k], dtype=np.int64), name="flat_shape")
        initializers.append(flat_shape)
        nodes.append(oh.make_node("Reshape", ["top_ids", "flat_shape"], ["flat_ids"]))

        # ----- Reshape weights to (T*k, 1) --------------------------------
        w_shape = onh.from_array(np.array([T * k, 1], dtype=np.int64), name="w_shape")
        initializers.append(w_shape)
        nodes.append(oh.make_node("Reshape", ["top_weights", "w_shape"], ["routing_w"]))

        # ----- Gather FC1 weights (E, inter, H) by flat_ids -----------------
        initializers.append(
            onh.from_array(
                np.random.randn(E, inter, H).astype(np.float32), name="fc1_experts_weights"
            )
        )
        nodes.append(
            oh.make_node("Gather", ["fc1_experts_weights", "flat_ids"], ["sel_fc1_w"], axis=0)
        )
        nodes.append(oh.make_node("Transpose", ["sel_fc1_w"], ["sel_fc1_w_t"], perm=[0, 2, 1]))

        # ----- Unsqueeze input (T*k, 1, H) --------------------------------
        unsq_axis1 = onh.from_array(np.array([1], dtype=np.int64), name="unsq_axis1")
        initializers.append(unsq_axis1)
        nodes.append(oh.make_node("Unsqueeze", ["input", "unsq_axis1"], ["input_3d"]))

        # ----- FC1 MatMul: (T*k, 1, H) x (T*k, H, inter) -> (T*k, 1, inter) ---
        nodes.append(oh.make_node("MatMul", ["input_3d", "sel_fc1_w_t"], ["fc1_3d"]))
        sq_axis1 = onh.from_array(np.array([1], dtype=np.int64), name="sq_axis1")
        initializers.append(sq_axis1)
        nodes.append(oh.make_node("Squeeze", ["fc1_3d", "sq_axis1"], ["fc1_out"]))

        fc1_act_input = "fc1_out"
        if with_fc1_bias:
            initializers.append(
                onh.from_array(np.zeros((E, inter), dtype=np.float32), name="fc1_experts_bias")
            )
            nodes.append(
                oh.make_node("Gather", ["fc1_experts_bias", "flat_ids"], ["sel_fc1_b"], axis=0)
            )
            nodes.append(oh.make_node("Add", ["fc1_out", "sel_fc1_b"], ["fc1_biased"]))
            fc1_act_input = "fc1_biased"

        # ----- Activation -------------------------------------------------
        nodes.append(oh.make_node(activation, [fc1_act_input], ["fc1_act"]))

        # ----- Unsqueeze for FC2 (T*k, 1, inter) ----------------------------
        unsq_axis2 = onh.from_array(np.array([1], dtype=np.int64), name="unsq_axis2")
        initializers.append(unsq_axis2)
        nodes.append(oh.make_node("Unsqueeze", ["fc1_act", "unsq_axis2"], ["fc1_act_3d"]))

        # ----- Gather FC2 weights (E, H, inter) by flat_ids -----------------
        initializers.append(
            onh.from_array(
                np.random.randn(E, H, inter).astype(np.float32), name="fc2_experts_weights"
            )
        )
        nodes.append(
            oh.make_node("Gather", ["fc2_experts_weights", "flat_ids"], ["sel_fc2_w"], axis=0)
        )
        nodes.append(oh.make_node("Transpose", ["sel_fc2_w"], ["sel_fc2_w_t"], perm=[0, 2, 1]))

        # ----- FC2 MatMul: (T*k, 1, inter) x (T*k, inter, H) -> (T*k, 1, H) --
        nodes.append(oh.make_node("MatMul", ["fc1_act_3d", "sel_fc2_w_t"], ["fc2_3d"]))
        sq_axis2 = onh.from_array(np.array([1], dtype=np.int64), name="sq_axis2")
        initializers.append(sq_axis2)
        nodes.append(oh.make_node("Squeeze", ["fc2_3d", "sq_axis2"], ["fc2_out"]))

        fc2_final_input = "fc2_out"
        if with_fc2_bias:
            initializers.append(
                onh.from_array(np.zeros((E, H), dtype=np.float32), name="fc2_experts_bias")
            )
            nodes.append(
                oh.make_node("Gather", ["fc2_experts_bias", "flat_ids"], ["sel_fc2_b"], axis=0)
            )
            nodes.append(oh.make_node("Add", ["fc2_out", "sel_fc2_b"], ["fc2_biased"]))
            fc2_final_input = "fc2_biased"

        # ----- Weighted sum: (T*k, H) * (T*k, 1) -------------------------
        nodes.append(oh.make_node("Mul", [fc2_final_input, "routing_w"], ["output"]))

        # router_logits has shape (T, E) — raw routing scores; Softmax applied above.
        # input has shape (T*k, H) — one row per dispatched (token, expert) pair.
        graph_inputs = [
            oh.make_tensor_value_info("input", TFLOAT, [T * k, H]),
            oh.make_tensor_value_info("router_logits", TFLOAT, [T, E]),
        ]
        graph_outputs = [oh.make_tensor_value_info("output", TFLOAT, [T * k, H])]

        return oh.make_model(
            oh.make_graph(nodes, "moe_pattern", graph_inputs, graph_outputs, initializers),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

    def _moe_check_ort(self, model: "ModelProto", opt_onx: "ModelProto", feeds: dict) -> None:
        """Runs pre- and post-fusion models with OnnxRuntime and compares outputs."""
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)
        opt_ref = InferenceSession(
            opt_onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_moe_pattern_in_list(self):
        """MoEPattern must appear in the default ORT pattern list."""
        from yobx.xoptim.patterns_ort import get_onnxruntime_patterns

        patterns = get_onnxruntime_patterns()
        names = [p.__class__.__name__ for p in patterns]
        self.assertIn("MoEPattern", names)

    def test_moe_pattern_basic_relu_with_biases(self):
        """TopK + expert-gather + FC1+Relu+FC2 with both biases fuses to MoE."""
        np.random.seed(0)
        T, H, E, k = 4, 8, 4, 1
        model = self._make_moe_model(activation="Relu", with_fc1_bias=True, with_fc2_bias=True)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["MoE"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("MoE", op_types)
        self.assertNotIn("TopK", op_types)
        self.assertNotIn("Gather", op_types)
        self.assertNotIn("MatMul", op_types)
        fused = [n for n in opt_onx.graph.node if n.op_type == "MoE"]
        self.assertEqual(1, len(fused))
        self.assertEqual("com.microsoft", fused[0].domain)
        # Check the k attribute.
        k_attr = next((a for a in fused[0].attribute if a.name == "k"), None)
        self.assertIsNotNone(k_attr)
        self.assertEqual(1, k_attr.i)
        # Check activation_type attribute.
        act_attr = next((a for a in fused[0].attribute if a.name == "activation_type"), None)
        self.assertIsNotNone(act_attr)
        self.assertEqual("relu", act_attr.s.decode())
        feeds = {
            "input": np.random.randn(T * k, H).astype(np.float32),
            "router_logits": np.random.randn(T, E).astype(np.float32),
        }
        self._moe_check_ort(model, opt_onx, feeds)

    def test_moe_pattern_no_biases(self):
        """MoE pattern also fuses when both FC biases are absent."""
        np.random.seed(1)
        T, H, E, k = 4, 8, 4, 1
        model = self._make_moe_model(activation="Relu", with_fc1_bias=False, with_fc2_bias=False)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["MoE"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("MoE", op_types)
        feeds = {
            "input": np.random.randn(T * k, H).astype(np.float32),
            "router_logits": np.random.randn(T, E).astype(np.float32),
        }
        self._moe_check_ort(model, opt_onx, feeds)

    def test_moe_pattern_no_fc2_bias(self):
        """MoE pattern fuses when only FC1 bias is present."""
        np.random.seed(2)
        T, H, E, k = 4, 8, 4, 1
        model = self._make_moe_model(activation="Relu", with_fc1_bias=True, with_fc2_bias=False)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["MoE"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("MoE", op_types)
        feeds = {
            "input": np.random.randn(T * k, H).astype(np.float32),
            "router_logits": np.random.randn(T, E).astype(np.float32),
        }
        self._moe_check_ort(model, opt_onx, feeds)

    def test_moe_pattern_reshape_minus_one(self):
        """MoE pattern fuses when the index Reshape uses (-1,) instead of (T,)."""
        np.random.seed(3)
        T, H, inter, E, k = 4, 8, 16, 4, 1

        nodes = []
        initializers = []

        # Include Softmax before TopK (same as _make_moe_model) so that the
        # fused com.microsoft.MoE node (which applies Softmax internally) is
        # numerically equivalent to the pre-fusion subgraph.
        nodes.append(oh.make_node("Softmax", ["router_logits"], ["router_probs"], axis=-1))

        topk_k = onh.from_array(np.array([k], dtype=np.int64), name="topk_k")
        initializers.append(topk_k)
        nodes.append(
            oh.make_node("TopK", ["router_probs", "topk_k"], ["top_weights", "top_ids"], axis=1)
        )

        # Use -1 instead of T*k to flatten the indices.
        flat_shape = onh.from_array(np.array([-1], dtype=np.int64), name="flat_shape")
        initializers.append(flat_shape)
        nodes.append(oh.make_node("Reshape", ["top_ids", "flat_shape"], ["flat_ids"]))

        w_shape = onh.from_array(np.array([T * k, 1], dtype=np.int64), name="w_shape")
        initializers.append(w_shape)
        nodes.append(oh.make_node("Reshape", ["top_weights", "w_shape"], ["routing_w"]))

        initializers.append(
            onh.from_array(
                np.random.randn(E, inter, H).astype(np.float32), name="fc1_experts_weights"
            )
        )
        nodes.append(
            oh.make_node("Gather", ["fc1_experts_weights", "flat_ids"], ["sel_fc1_w"], axis=0)
        )
        nodes.append(oh.make_node("Transpose", ["sel_fc1_w"], ["sel_fc1_w_t"], perm=[0, 2, 1]))

        unsq_axis1 = onh.from_array(np.array([1], dtype=np.int64), name="unsq_axis1")
        initializers.append(unsq_axis1)
        nodes.append(oh.make_node("Unsqueeze", ["input", "unsq_axis1"], ["input_3d"]))

        nodes.append(oh.make_node("MatMul", ["input_3d", "sel_fc1_w_t"], ["fc1_3d"]))
        sq_axis1 = onh.from_array(np.array([1], dtype=np.int64), name="sq_axis1")
        initializers.append(sq_axis1)
        nodes.append(oh.make_node("Squeeze", ["fc1_3d", "sq_axis1"], ["fc1_out"]))
        nodes.append(oh.make_node("Relu", ["fc1_out"], ["fc1_act"]))

        unsq_axis2 = onh.from_array(np.array([1], dtype=np.int64), name="unsq_axis2")
        initializers.append(unsq_axis2)
        nodes.append(oh.make_node("Unsqueeze", ["fc1_act", "unsq_axis2"], ["fc1_act_3d"]))

        initializers.append(
            onh.from_array(
                np.random.randn(E, H, inter).astype(np.float32), name="fc2_experts_weights"
            )
        )
        nodes.append(
            oh.make_node("Gather", ["fc2_experts_weights", "flat_ids"], ["sel_fc2_w"], axis=0)
        )
        nodes.append(oh.make_node("Transpose", ["sel_fc2_w"], ["sel_fc2_w_t"], perm=[0, 2, 1]))

        nodes.append(oh.make_node("MatMul", ["fc1_act_3d", "sel_fc2_w_t"], ["fc2_3d"]))
        sq_axis2 = onh.from_array(np.array([1], dtype=np.int64), name="sq_axis2")
        initializers.append(sq_axis2)
        nodes.append(oh.make_node("Squeeze", ["fc2_3d", "sq_axis2"], ["fc2_out"]))

        nodes.append(oh.make_node("Mul", ["fc2_out", "routing_w"], ["output"]))

        model = oh.make_model(
            oh.make_graph(
                nodes,
                "moe_minus_one",
                [
                    oh.make_tensor_value_info("input", TFLOAT, [T * k, H]),
                    oh.make_tensor_value_info("router_logits", TFLOAT, [T, E]),
                ],
                [oh.make_tensor_value_info("output", TFLOAT, [T * k, H])],
                initializers,
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["MoE"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("MoE", [n.op_type for n in opt_onx.graph.node])
        feeds = {
            "input": np.random.randn(T * k, H).astype(np.float32),
            "router_logits": np.random.randn(T, E).astype(np.float32),
        }
        self._moe_check_ort(model, opt_onx, feeds)

    def test_moe_pattern_shape_inference(self):
        """Output of the fused MoE node has the same shape as the input."""
        np.random.seed(4)
        T, H, E, k = 6, 8, 4, 1
        model = self._make_moe_model(
            num_tokens=T, hidden_size=H, activation="Relu", with_fc1_bias=True, with_fc2_bias=True
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["MoE"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        # The graph output shape must be preserved.
        out_info = opt_onx.graph.output[0]
        dims = [d.dim_value for d in out_info.type.tensor_type.shape.dim]
        self.assertEqual([T * k, H], dims)
        feeds = {
            "input": np.random.randn(T * k, H).astype(np.float32),
            "router_logits": np.random.randn(T, E).astype(np.float32),
        }
        self._moe_check_ort(model, opt_onx, feeds)

    def test_moe_pattern_top_k_2(self):
        """MoE pattern fuses correctly when k=2 (each token dispatched to 2 experts)."""
        np.random.seed(5)
        T, H, E, k = 4, 8, 4, 2
        model = self._make_moe_model(
            num_tokens=T,
            hidden_size=H,
            num_experts=E,
            top_k=k,
            activation="Relu",
            with_fc1_bias=True,
            with_fc2_bias=True,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["MoE"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("MoE", [n.op_type for n in opt_onx.graph.node])
        fused = [n for n in opt_onx.graph.node if n.op_type == "MoE"]
        k_attr = next((a for a in fused[0].attribute if a.name == "k"), None)
        self.assertIsNotNone(k_attr)
        self.assertEqual(2, k_attr.i)
        # Output shape should be (T*k, H) = (8, H).
        out_info = opt_onx.graph.output[0]
        dims = [d.dim_value for d in out_info.type.tensor_type.shape.dim]
        self.assertEqual([T * k, H], dims)
        # ORT's com.microsoft.MoE op expects router_logits to have the same number
        # of rows as input (T*k), whereas this test's pre-fusion topology uses T
        # rows for router_logits.  Numerical equivalence is therefore only verified
        # for k=1 where T*k == T; here we check structural correctness only.

    def test_moe_pattern_no_match_missing_topk(self):
        """Pattern must NOT fire when there is no TopK feeding the routing weight."""
        np.random.seed(6)
        num_tokens, hidden_size, num_experts = 4, 8, 4
        # Replace TopK with a direct Softmax output (no actual TopK in graph).
        nodes = [
            oh.make_node("Softmax", ["router_probs"], ["routing_w"], axis=-1),
            oh.make_node("Relu", ["input"], ["output"]),
            oh.make_node("Mul", ["output", "routing_w"], ["final"]),
        ]
        model = oh.make_model(
            oh.make_graph(
                nodes,
                "no_topk",
                [
                    oh.make_tensor_value_info("input", TFLOAT, [num_tokens, hidden_size]),
                    oh.make_tensor_value_info("router_probs", TFLOAT, [num_tokens, num_experts]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, [num_tokens, hidden_size])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["MoE"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("MoE", [n.op_type for n in opt_onx.graph.node])

    # -----------------------------------------------------------------------
    # LinearAttention tests
    # -----------------------------------------------------------------------

    def _make_linear_attention_model(
        self,
        batch: int = 2,
        q_num_heads: int = 4,
        kv_num_heads: int = 2,
        head_dim_k: int = 8,
        head_dim_v: int = 8,
        update_rule: str = "linear",
    ) -> "ModelProto":
        """Builds a pre-fusion linear attention ONNX subgraph (T=1, decoding).

        The graph implements (``'linear'`` rule):

        1. Unpack 3-D packed Q/K/V:
           ``Reshape + Transpose + Squeeze``
        2. Outer product: ``k ⊗ v  = Mul(Unsqueeze(k, -1), Unsqueeze(v, -2))``
        3. State update: ``S_new = Add(past_state, kv)``
        4. Output: ``out = Mul(Squeeze(MatMul(Unsqueeze(q, -2), S_new)), scale)``
        5. Repack: ``Unsqueeze + Transpose + Reshape``

        For ``update_rule='gated'`` an ``Exp + Mul`` decay path is inserted
        before the ``Add``.
        """
        B = batch
        Hq, Hkv, Dk, Dv = q_num_heads, kv_num_heads, head_dim_k, head_dim_v
        T = 1  # decoding step

        import math

        scale = 1.0 / math.sqrt(Dk)

        nodes = []
        initializers = []

        # ---- reshape shapes -----------------------------------------------
        shape_q = onh.from_array(np.array([0, T, Hq, Dk], dtype=np.int64), name="shape_q")
        shape_k = onh.from_array(np.array([0, T, Hkv, Dk], dtype=np.int64), name="shape_k")
        shape_v = onh.from_array(np.array([0, T, Hkv, Dv], dtype=np.int64), name="shape_v")
        shape_out = onh.from_array(np.array([0, -1, Hq * Dv], dtype=np.int64), name="shape_out")
        initializers += [shape_q, shape_k, shape_v, shape_out]

        # ---- Unsqueeze / Squeeze axes ------------------------------------
        ax2 = onh.from_array(np.array([2], dtype=np.int64), name="ax2")
        ax_m1 = onh.from_array(np.array([-1], dtype=np.int64), name="ax_m1")
        ax_m2 = onh.from_array(np.array([-2], dtype=np.int64), name="ax_m2")
        initializers += [ax2, ax_m1, ax_m2]

        # ---- scale constant -----------------------------------------------
        scale_cst = onh.from_array(np.array([scale], dtype=np.float32), name="scale_cst")
        initializers.append(scale_cst)

        # ---- unpack Q -------------------------------------------------------
        nodes.append(oh.make_node("Reshape", ["query", "shape_q"], ["q_4d"]))
        nodes.append(oh.make_node("Transpose", ["q_4d"], ["q_t"], perm=[0, 2, 1, 3]))
        nodes.append(oh.make_node("Squeeze", ["q_t", "ax2"], ["q_sq"]))

        # ---- unpack K -------------------------------------------------------
        nodes.append(oh.make_node("Reshape", ["key", "shape_k"], ["k_4d"]))
        nodes.append(oh.make_node("Transpose", ["k_4d"], ["k_t"], perm=[0, 2, 1, 3]))
        nodes.append(oh.make_node("Squeeze", ["k_t", "ax2"], ["k_sq"]))

        # ---- unpack V -------------------------------------------------------
        nodes.append(oh.make_node("Reshape", ["value", "shape_v"], ["v_4d"]))
        nodes.append(oh.make_node("Transpose", ["v_4d"], ["v_t"], perm=[0, 2, 1, 3]))
        nodes.append(oh.make_node("Squeeze", ["v_t", "ax2"], ["v_sq"]))

        # ---- outer product k ⊗ v ------------------------------------------
        nodes.append(oh.make_node("Unsqueeze", ["k_sq", "ax_m1"], ["k_col"]))
        nodes.append(oh.make_node("Unsqueeze", ["v_sq", "ax_m2"], ["v_row"]))
        nodes.append(oh.make_node("Mul", ["k_col", "v_row"], ["kv_outer"]))

        # ---- state update -------------------------------------------------
        if update_rule == "linear":
            nodes.append(oh.make_node("Add", ["past_state", "kv_outer"], ["new_state"]))
        elif update_rule == "gated":
            # Gated: S_new = Mul(Exp(decay_sq), past_state) + kv_outer
            nodes.append(oh.make_node("Reshape", ["decay", "shape_k"], ["d_4d"]))
            nodes.append(oh.make_node("Transpose", ["d_4d"], ["d_t"], perm=[0, 2, 1, 3]))
            nodes.append(oh.make_node("Squeeze", ["d_t", "ax2"], ["d_sq"]))
            nodes.append(oh.make_node("Exp", ["d_sq"], ["d_exp"]))
            # Broadcast Exp(decay) over d_v dimension:
            # d_exp: [B, Hkv, Dk] → need [B, Hkv, Dk, 1] * past_state [B, Hkv, Dk, Dv]
            nodes.append(oh.make_node("Unsqueeze", ["d_exp", "ax_m1"], ["d_exp_col"]))
            nodes.append(oh.make_node("Mul", ["d_exp_col", "past_state"], ["s_gated"]))
            nodes.append(oh.make_node("Add", ["s_gated", "kv_outer"], ["new_state"]))
        else:
            raise ValueError(f"Unknown update_rule: {update_rule}")

        # ---- output computation -------------------------------------------
        nodes.append(oh.make_node("Unsqueeze", ["q_sq", "ax_m2"], ["q_row"]))
        nodes.append(oh.make_node("MatMul", ["q_row", "new_state"], ["out_us"]))
        nodes.append(oh.make_node("Squeeze", ["out_us", "ax_m2"], ["out_sq"]))
        nodes.append(oh.make_node("Mul", ["out_sq", "scale_cst"], ["out_scaled"]))

        # ---- repack to 3D -------------------------------------------------
        nodes.append(oh.make_node("Unsqueeze", ["out_scaled", "ax2"], ["out_4d"]))
        nodes.append(oh.make_node("Transpose", ["out_4d"], ["out_t"], perm=[0, 2, 1, 3]))
        nodes.append(oh.make_node("Reshape", ["out_t", "shape_out"], ["output"]))

        # ---- graph inputs / outputs ---------------------------------------
        graph_inputs = [
            oh.make_tensor_value_info("query", TFLOAT, [B, T, Hq * Dk]),
            oh.make_tensor_value_info("key", TFLOAT, [B, T, Hkv * Dk]),
            oh.make_tensor_value_info("value", TFLOAT, [B, T, Hkv * Dv]),
            oh.make_tensor_value_info("past_state", TFLOAT, [B, Hkv, Dk, Dv]),
        ]
        if update_rule == "gated":
            graph_inputs.append(oh.make_tensor_value_info("decay", TFLOAT, [B, T, Hkv * Dk]))

        graph_outputs = [
            oh.make_tensor_value_info("output", TFLOAT, [B, T, Hq * Dv]),
            oh.make_tensor_value_info("new_state", TFLOAT, [B, Hkv, Dk, Dv]),
        ]

        return oh.make_model(
            oh.make_graph(nodes, "linear_attention", graph_inputs, graph_outputs, initializers),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

    def test_linear_attention_in_list(self):
        """LinearAttentionPattern must appear in the default ORT pattern list."""
        from yobx.xoptim.patterns_ort import get_onnxruntime_patterns

        patterns = get_onnxruntime_patterns()
        names = [p.__class__.__name__ for p in patterns]
        self.assertIn("LinearAttentionPattern", names)

    def test_linear_attention_pattern_linear_rule(self):
        """Linear rule (no decay): fuses to com.microsoft.LinearAttention."""
        np.random.seed(0)
        model = self._make_linear_attention_model(update_rule="linear")
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["LinearAttention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("LinearAttention", op_types)
        self.assertNotIn("MatMul", op_types)
        self.assertNotIn("Unsqueeze", op_types)
        fused = [n for n in opt_onx.graph.node if n.op_type == "LinearAttention"]
        self.assertEqual(1, len(fused))
        self.assertEqual("com.microsoft", fused[0].domain)
        # Check update_rule attribute
        rule_attr = next((a for a in fused[0].attribute if a.name == "update_rule"), None)
        self.assertIsNotNone(rule_attr)
        self.assertEqual("linear", rule_attr.s.decode())
        # Verify the state output is preserved in the fused graph
        output_names = [o.name for o in opt_onx.graph.output]
        self.assertIn("new_state", output_names)

    def test_linear_attention_pattern_gated_rule(self):
        """Gated rule (with Exp(decay)): fuses to com.microsoft.LinearAttention."""
        np.random.seed(1)
        model = self._make_linear_attention_model(update_rule="gated")
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["LinearAttention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("LinearAttention", op_types)
        self.assertNotIn("Exp", op_types)
        self.assertNotIn("MatMul", op_types)
        fused = [n for n in opt_onx.graph.node if n.op_type == "LinearAttention"]
        self.assertEqual(1, len(fused))
        self.assertEqual("com.microsoft", fused[0].domain)
        rule_attr = next((a for a in fused[0].attribute if a.name == "update_rule"), None)
        self.assertIsNotNone(rule_attr)
        self.assertEqual("gated", rule_attr.s.decode())

    def test_linear_attention_shape_inference(self):
        """Output of fused LinearAttention node has the correct shape."""
        np.random.seed(2)
        B, Hq, Hkv, Dk, Dv = 2, 4, 2, 8, 8
        model = self._make_linear_attention_model(
            batch=B,
            q_num_heads=Hq,
            kv_num_heads=Hkv,
            head_dim_k=Dk,
            head_dim_v=Dv,
            update_rule="linear",
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["LinearAttention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        fused = [n for n in opt_onx.graph.node if n.op_type == "LinearAttention"]
        self.assertEqual(1, len(fused))
        # Check q_num_heads / kv_num_heads attributes
        qnh = next((a for a in fused[0].attribute if a.name == "q_num_heads"), None)
        kvnh = next((a for a in fused[0].attribute if a.name == "kv_num_heads"), None)
        self.assertIsNotNone(qnh)
        self.assertIsNotNone(kvnh)
        self.assertEqual(Hq, qnh.i)
        self.assertEqual(Hkv, kvnh.i)

    def test_linear_attention_no_match_without_unpack(self):
        """Pattern must NOT match when the Q/K/V are already 3D without unpack."""
        import math

        B, Hq, Hkv, Dk, Dv = 2, 2, 2, 4, 4
        scale = 1.0 / math.sqrt(Dk)
        # Build a simplified graph missing the Reshape+Transpose+Squeeze unpack
        nodes = []
        initializers = []
        ax_m1 = onh.from_array(np.array([-1], dtype=np.int64), name="ax_m1")
        ax_m2 = onh.from_array(np.array([-2], dtype=np.int64), name="ax_m2")
        scale_cst = onh.from_array(np.array([scale], dtype=np.float32), name="scale_cst")
        initializers += [ax_m1, ax_m2, scale_cst]

        # Outer product directly on raw k/v (no unpack)
        nodes.append(oh.make_node("Unsqueeze", ["key", "ax_m1"], ["k_col"]))
        nodes.append(oh.make_node("Unsqueeze", ["value", "ax_m2"], ["v_row"]))
        nodes.append(oh.make_node("Mul", ["k_col", "v_row"], ["kv_outer"]))
        nodes.append(oh.make_node("Add", ["past_state", "kv_outer"], ["new_state"]))
        nodes.append(oh.make_node("Unsqueeze", ["query", "ax_m2"], ["q_row"]))
        nodes.append(oh.make_node("MatMul", ["q_row", "new_state"], ["out_us"]))
        nodes.append(oh.make_node("Squeeze", ["out_us", "ax_m2"], ["out_sq"]))
        nodes.append(oh.make_node("Mul", ["out_sq", "scale_cst"], ["output"]))

        graph_inputs = [
            oh.make_tensor_value_info("query", TFLOAT, [B, Hq, Dk]),
            oh.make_tensor_value_info("key", TFLOAT, [B, Hkv, Dk]),
            oh.make_tensor_value_info("value", TFLOAT, [B, Hkv, Dv]),
            oh.make_tensor_value_info("past_state", TFLOAT, [B, Hkv, Dk, Dv]),
        ]
        graph_outputs = [oh.make_tensor_value_info("output", TFLOAT, [B, Hq, Dv])]
        model = oh.make_model(
            oh.make_graph(nodes, "no_unpack", graph_inputs, graph_outputs, initializers),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["LinearAttention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        # Without the 3D unpack path the pattern should NOT fire.
        self.assertNotIn("LinearAttention", [n.op_type for n in opt_onx.graph.node])

    # ------------------------------------------------------------------
    # DecoderAttentionPattern tests
    # ------------------------------------------------------------------

    def make_decoder_attention_model(
        self,
        seq_len: int = 3,
        enc_seq_len: int = 5,
        batch: int = 2,
        num_heads: int = 4,
        head_dim: int = 8,
        cross_attn: bool = True,
    ) -> "ModelProto":
        """Builds a pre-fusion seq-first decoder attention ONNX subgraph.

        For ``cross_attn=True``: Q comes from *query* ``(S, B, H)``, K and V
        come from *key* ``(T, B, H)``.

        For ``cross_attn=False``: Q, K, V all come from *query* (self-attention).

        The graph implements the standard multi-head attention computation in
        sequence-first format using the transposes expected by
        ``com.microsoft.DecoderAttention``.
        """
        import math

        S = seq_len
        T = enc_seq_len if cross_attn else seq_len
        B = batch
        N = num_heads
        d = head_dim
        H = N * d
        scale = 1.0 / math.sqrt(d)

        nodes = []
        initializers = []

        # Weight / bias initializers
        q_weight = onh.from_array(np.random.randn(H, H).astype(np.float32), name="q_weight")
        k_weight = onh.from_array(np.random.randn(H, H).astype(np.float32), name="k_weight")
        v_weight = onh.from_array(np.random.randn(H, H).astype(np.float32), name="v_weight")
        q_bias = onh.from_array(np.random.randn(H).astype(np.float32), name="q_bias")
        k_bias = onh.from_array(np.random.randn(H).astype(np.float32), name="k_bias")
        v_bias = onh.from_array(np.random.randn(H).astype(np.float32), name="v_bias")
        scale_cst = onh.from_array(np.array([scale], dtype=np.float32), name="scale")
        shape_4d = onh.from_array(np.array([0, 0, N, d], dtype=np.int64), name="shape_4d")
        shape_out = onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="shape_out")
        initializers += [
            q_weight,
            k_weight,
            v_weight,
            q_bias,
            k_bias,
            v_bias,
            scale_cst,
            shape_4d,
            shape_out,
        ]

        key_input = "key" if cross_attn else "query"

        # Q branch: (S,B,H) → MatMul → Add → Reshape → Transpose([1,2,0,3])
        nodes += [
            oh.make_node("MatMul", ["query", "q_weight"], ["mm_q"]),
            oh.make_node("Add", ["mm_q", "q_bias"], ["add_q"]),
            oh.make_node("Reshape", ["add_q", "shape_4d"], ["re_q"]),
            oh.make_node("Transpose", ["re_q"], ["tr_q"], perm=[1, 2, 0, 3]),
        ]

        # K branch: (T,B,H) → MatMul → Add → Reshape → Transpose → Transpose([0,1,3,2])
        nodes += [
            oh.make_node("MatMul", [key_input, "k_weight"], ["mm_k"]),
            oh.make_node("Add", ["mm_k", "k_bias"], ["add_k"]),
            oh.make_node("Reshape", ["add_k", "shape_4d"], ["re_k"]),
            oh.make_node("Transpose", ["re_k"], ["tr_k"], perm=[1, 2, 0, 3]),
            oh.make_node("Transpose", ["tr_k"], ["tr_kt"], perm=[0, 1, 3, 2]),
        ]

        # V branch: (T,B,H) → MatMul → Add → Reshape → Transpose([1,2,0,3])
        nodes += [
            oh.make_node("MatMul", [key_input, "v_weight"], ["mm_v"]),
            oh.make_node("Add", ["mm_v", "v_bias"], ["add_v"]),
            oh.make_node("Reshape", ["add_v", "shape_4d"], ["re_v"]),
            oh.make_node("Transpose", ["re_v"], ["tr_v"], perm=[1, 2, 0, 3]),
        ]

        # Attention computation
        nodes += [
            oh.make_node("Mul", ["tr_q", "scale"], ["q_scaled"]),
            oh.make_node("MatMul", ["q_scaled", "tr_kt"], ["attn_logits"]),
            oh.make_node("Softmax", ["attn_logits"], ["attn_probs"], axis=-1),
        ]

        attn_probs_out = "attn_probs"
        nodes += [oh.make_node("MatMul", [attn_probs_out, "tr_v"], ["attn_out"])]

        # Output: (B,N,S,d) → Transpose([2,0,1,3]) → (S,B,N,d) → Reshape
        nodes += [
            oh.make_node("Transpose", ["attn_out"], ["tr_out"], perm=[2, 0, 1, 3]),
            oh.make_node("Reshape", ["tr_out", "shape_out"], ["output"]),
        ]

        # Graph I/O
        graph_inputs = [oh.make_tensor_value_info("query", TFLOAT, [S, B, H])]
        if cross_attn:
            graph_inputs.append(oh.make_tensor_value_info("key", TFLOAT, [T, B, H]))
        graph_outputs = [oh.make_tensor_value_info("output", TFLOAT, [S, B, H])]

        return oh.make_model(
            oh.make_graph(nodes, "decoder_attention", graph_inputs, graph_outputs, initializers),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

    def test_decoder_attention_in_pattern_list(self):
        """DecoderAttentionPattern must appear in the default ORT pattern list."""
        from yobx.xoptim.patterns_ort import get_onnxruntime_patterns

        patterns = get_onnxruntime_patterns()
        names = [p.__class__.__name__ for p in patterns]
        self.assertIn("DecoderAttentionPattern", names)

    def test_decoder_attention_pattern_cross_attention(self):
        """Cross-attention fuses into com.microsoft.DecoderAttention with static_kv=True."""
        np.random.seed(0)
        model = self.make_decoder_attention_model(cross_attn=True)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["DecoderAttention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("DecoderAttention", op_types)
        self.assertNotIn("MatMul", op_types)
        self.assertNotIn("Softmax", op_types)

        fused = [n for n in opt_onx.graph.node if n.op_type == "DecoderAttention"]
        self.assertEqual(1, len(fused))
        self.assertEqual("com.microsoft", fused[0].domain)

        # num_heads attribute
        nh = next((a for a in fused[0].attribute if a.name == "num_heads"), None)
        self.assertIsNotNone(nh)
        self.assertEqual(4, nh.i)

        # static_kv should be True for cross-attention
        from onnx import numpy_helper as _onh

        init_map = {i.name: _onh.to_array(i) for i in opt_onx.graph.initializer}
        static_kv_val = init_map.get(fused[0].input[8])
        self.assertIsNotNone(static_kv_val)
        self.assertTrue(bool(static_kv_val.flat[0]))

    def test_decoder_attention_pattern_self_attention(self):
        """Self-attention fuses into com.microsoft.DecoderAttention with static_kv=False."""
        np.random.seed(1)
        model = self.make_decoder_attention_model(cross_attn=False)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["DecoderAttention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertIn("DecoderAttention", op_types)

        fused = [n for n in opt_onx.graph.node if n.op_type == "DecoderAttention"]
        self.assertEqual(1, len(fused))

        # static_kv should be False for self-attention
        from onnx import numpy_helper as _onh

        init_map = {i.name: _onh.to_array(i) for i in opt_onx.graph.initializer}
        static_kv_val = init_map.get(fused[0].input[8])
        self.assertIsNotNone(static_kv_val)
        self.assertFalse(bool(static_kv_val.flat[0]))

        # For self-attention query == key input
        self.assertEqual(fused[0].input[0], fused[0].input[1])

    def test_decoder_attention_shape_inference(self):
        """Fused DecoderAttention has the correct output shape (S, B, H)."""
        np.random.seed(2)
        S, T, B, N, d = 3, 5, 2, 4, 8
        H = N * d
        model = self.make_decoder_attention_model(
            seq_len=S, enc_seq_len=T, batch=B, num_heads=N, head_dim=d, cross_attn=True
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["DecoderAttention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        fused = [n for n in opt_onx.graph.node if n.op_type == "DecoderAttention"]
        self.assertEqual(1, len(fused))

        # Graph output shape should be (S, B, H)
        out_shape = [dim.dim_value for dim in opt_onx.graph.output[0].type.tensor_type.shape.dim]
        self.assertEqual([S, B, H], out_shape)

    def test_decoder_attention_kv_weight_shape(self):
        """kv_weight produced by the fusion has shape (H, 2*H)."""
        np.random.seed(3)
        N, d = 4, 8
        H = N * d
        model = self.make_decoder_attention_model(num_heads=N, head_dim=d, cross_attn=True)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["DecoderAttention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        fused = [n for n in opt_onx.graph.node if n.op_type == "DecoderAttention"]
        self.assertEqual(1, len(fused))

        # The kv_weight initializer is input[3] of the fused node.
        kv_weight_name = fused[0].input[3]
        from onnx import numpy_helper as _onh

        init_map = {i.name: _onh.to_array(i) for i in opt_onx.graph.initializer}
        kv_weight = init_map.get(kv_weight_name)
        self.assertIsNotNone(kv_weight)
        self.assertEqual((H, 2 * H), kv_weight.shape)

    def test_decoder_attention_bias_shape(self):
        """Combined bias produced by the fusion has shape (3*H,)."""
        np.random.seed(4)
        N, d = 4, 8
        H = N * d
        model = self.make_decoder_attention_model(num_heads=N, head_dim=d, cross_attn=True)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["DecoderAttention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        fused = [n for n in opt_onx.graph.node if n.op_type == "DecoderAttention"]
        self.assertEqual(1, len(fused))

        bias_name = fused[0].input[4]
        from onnx import numpy_helper as _onh

        init_map = {i.name: _onh.to_array(i) for i in opt_onx.graph.initializer}
        bias = init_map.get(bias_name)
        self.assertIsNotNone(bias)
        self.assertEqual((3 * H,), bias.shape)

    def test_decoder_attention_no_match_wrong_transpose(self):
        """Pattern must NOT match when the output Transpose is batch-first [0,2,1,3]."""
        import math

        N, d, B, S = 4, 8, 2, 3
        H = N * d
        scale = 1.0 / math.sqrt(d)
        nodes = []
        initializers = []
        q_weight = onh.from_array(np.random.randn(H, H).astype(np.float32), name="q_weight")
        k_weight = onh.from_array(np.random.randn(H, H).astype(np.float32), name="k_weight")
        v_weight = onh.from_array(np.random.randn(H, H).astype(np.float32), name="v_weight")
        scale_cst = onh.from_array(np.array([scale], dtype=np.float32), name="scale")
        shape_4d = onh.from_array(np.array([0, 0, N, d], dtype=np.int64), name="shape_4d")
        shape_out = onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="shape_out")
        initializers += [q_weight, k_weight, v_weight, scale_cst, shape_4d, shape_out]

        key_input = "key"
        nodes += [
            oh.make_node("MatMul", ["query", "q_weight"], ["mm_q"]),
            oh.make_node("Reshape", ["mm_q", "shape_4d"], ["re_q"]),
            # Batch-first projection transpose perm=[0,2,1,3] instead of [1,2,0,3]
            oh.make_node("Transpose", ["re_q"], ["tr_q"], perm=[0, 2, 1, 3]),
            oh.make_node("MatMul", [key_input, "k_weight"], ["mm_k"]),
            oh.make_node("Reshape", ["mm_k", "shape_4d"], ["re_k"]),
            oh.make_node("Transpose", ["re_k"], ["tr_k"], perm=[0, 2, 1, 3]),
            oh.make_node("Transpose", ["tr_k"], ["tr_kt"], perm=[0, 1, 3, 2]),
            oh.make_node("MatMul", [key_input, "v_weight"], ["mm_v"]),
            oh.make_node("Reshape", ["mm_v", "shape_4d"], ["re_v"]),
            oh.make_node("Transpose", ["re_v"], ["tr_v"], perm=[0, 2, 1, 3]),
            oh.make_node("Mul", ["tr_q", "scale"], ["q_scaled"]),
            oh.make_node("MatMul", ["q_scaled", "tr_kt"], ["attn_logits"]),
            oh.make_node("Softmax", ["attn_logits"], ["attn_probs"], axis=-1),
            oh.make_node("MatMul", ["attn_probs", "tr_v"], ["attn_out"]),
            # Batch-first output transpose
            oh.make_node("Transpose", ["attn_out"], ["tr_out"], perm=[0, 2, 1, 3]),
            oh.make_node("Reshape", ["tr_out", "shape_out"], ["output"]),
        ]
        graph_inputs = [
            oh.make_tensor_value_info("query", TFLOAT, [B, S, H]),
            oh.make_tensor_value_info("key", TFLOAT, [B, S, H]),
        ]
        graph_outputs = [oh.make_tensor_value_info("output", TFLOAT, [B, S, H])]
        model = oh.make_model(
            oh.make_graph(nodes, "batch_first", graph_inputs, graph_outputs, initializers),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["DecoderAttention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertNotIn("DecoderAttention", [n.op_type for n in opt_onx.graph.node])


if __name__ == "__main__":
    unittest.main(verbosity=2)
