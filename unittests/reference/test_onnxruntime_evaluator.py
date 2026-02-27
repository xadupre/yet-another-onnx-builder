import unittest
from typing import Any, Dict, Optional, Tuple
import numpy as np
import ml_dtypes
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import torch
from yobx.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings, requires_cuda
from yobx.helpers.onnx_helper import tensor_dtype_to_np_dtype
from yobx.torch.torch_helper import onnx_dtype_to_torch_dtype
from yobx.reference._inference_session import _InferenceSession
from yobx.reference import ExtendedReferenceEvaluator, ReportResultComparison
from yobx.reference.onnxruntime_evaluator import OnnxruntimeEvaluator

TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64


class TestOnnxruntimeEvaluator(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    @ignore_warnings(FutureWarning)
    def test_ort_eval_scan_cdist_add(self):

        def dist(unused: torch.Tensor, x: torch.Tensor, samex: torch.Tensor):
            sub = samex - x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [unused.clone(), rd]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                z = torch.tensor([0], dtype=torch.float32)
                y = x.clone()
                out = torch.ops.higher_order.scan(dist, [z], [x], additional_inputs=[y])
                return out[1]

        x = torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32)
        model = ScanModel()
        _expected = model(x)
        self.skipTest("not implemented yet")
        """
        onx = to_onnx(
            model,
            (x,),
            optimize=True,
            export_options=ExportOptions(decomposition_table="default", strict=False),
            inline=False,
        )
        filename = self.get_dump_file("test_ort_eval_scan_cdist_add.onnx")
        onnx.save(onx, filename)
        inits = [i.name for i in onx.graph.initializer]
        self.assertEqual(inits, ["c_lifted_tensor_0"])
        name = onx.graph.input[0].name

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {name: x.numpy()})[0]
        self.assertEqualArray(expected, got)

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {name: x.numpy()})[0]
        self.assertEqualArray(expected, got)

        orte = OnnxruntimeEvaluator(onx)
        got = orte.run(None, {name: x.numpy()})[0]
        self.assertEqualArray(expected, got)
        """

    @ignore_warnings((UserWarning, FutureWarning))
    def test_ort_eval_cond(self):
        import torch

        class TwoInputs(torch.nn.Module):
            def forward(self, x, y):
                def true_fn(x, y):
                    return torch.sin(x), torch.cos(x) + y

                def false_fn(x, y):
                    return torch.cos(x), torch.sin(x) + y

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x, y])

        x, y = torch.rand(5, 3), torch.rand(5, 3)
        model = TwoInputs()
        self.skipTest("not implemented yet")
        onx = model, (x, y)  # to_onnx(model, (x, y), inline=False)
        self.assertEqual(len(onx.functions), 2)

        # ExtendedReferenceEvaluator
        ref = ExtendedReferenceEvaluator(onx)
        for _x in (x, -x):
            expected = model(_x, y)
            got = ref.run(None, {"x": _x.detach().numpy(), "y": y.detach().numpy()})
            self.assertEqual(len(expected), len(got))
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=1e-5)

        # OnnxruntimeEvaluator
        ref = OnnxruntimeEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x, y)
            got = ref.run(None, {"x": _x.detach().numpy(), "y": y.detach().numpy()})
            self.assertEqual(len(expected), len(got))
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=1e-5)

    def test_constant_bool(self):
        node = oh.make_node(
            "Constant",
            [],
            ["cbool"],
            value=onh.from_array(np.array(True, dtype=np.bool_)),
        )
        ref = ExtendedReferenceEvaluator(node)
        got = ref.run(None, {})[0]
        self.assertEqual(got.dtype, np.bool_)
        self.assertEqual(got, True)
        ref = OnnxruntimeEvaluator(node, opsets=21)
        got = ref.run(None, {})[0]
        self.assertEqual(len(ref._cache), 1)
        values = list(ref._cache.values())
        _, sess = values[0]
        got2 = sess.run(None, {})[0]
        self.assertIn(got2.dtype, (torch.bool, np.bool_))
        self.assertEqual(got2, True)

        self.assertIn(got.dtype, (torch.bool, np.bool_))
        self.assertEqual(got, True)

    def test_constant_bool_array(self):
        node = oh.make_node(
            "Constant",
            [],
            ["cbool"],
            value=onh.from_array(np.array([True], dtype=np.bool_)),
        )
        ref = ExtendedReferenceEvaluator(node)
        got = ref.run(None, {})[0]
        self.assertEqual(got.dtype, np.bool_)
        self.assertEqual(got[0], True)
        ref = OnnxruntimeEvaluator(node, opsets=21)
        got = ref.run(None, {})[0]
        self.assertEqual(len(ref._cache), 1)
        values = list(ref._cache.values())
        _, sess = values[0]
        got2 = sess.run(None, {})[0]
        self.assertIn(got2.dtype, (torch.bool, np.bool_))
        self.assertEqual(got2[0], True)

        self.assertIn(got.dtype, (torch.bool, np.bool_))
        self.assertEqual(got[0], True)

    def test_constant_bool_input(self):
        node = oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["bin"], ["bout"])],
                "test",
                [oh.make_tensor_value_info("bin", onnx.TensorProto.BOOL, [1])],
                [oh.make_tensor_value_info("bin", onnx.TensorProto.BOOL, [1])],
            ),
            ir_version=10,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        feeds = dict(bin=np.array([True], dtype=np.bool_))
        ref = ExtendedReferenceEvaluator(node)

        got = ref.run(None, feeds)[0]
        self.assertEqual(got.dtype, np.bool_)
        self.assertEqual(got[0], True)

        ref = OnnxruntimeEvaluator(node, opsets=21)
        got = ref.run(None, feeds)[0]
        self.assertEqual(got.dtype, np.bool_)
        self.assertEqual(got[0], True)

        feeds = dict(bin=torch.tensor([True], dtype=torch.bool))
        got = ref.run(None, feeds)[0]
        self.assertEqual(got.dtype, torch.bool)
        self.assertEqual(got[0], True)

    @hide_stdout()
    def test_ort_eval_loop(self):
        model = torch.nn.EmbeddingBag(num_embeddings=49157, embedding_dim=32, mode="sum")
        a = torch.tensor([[39906, 39906]]).long()
        example_args = (a,)
        model_eval = model.eval()
        expected = model(*example_args)

        self.skipTest("not implemented yet")
        onx = model_eval, example_args  # to_onnx(model_eval, example_args, optimize=True)
        self.assertIn("Loop", set(n.op_type for n in onx.graph.node))

        ref = OnnxruntimeEvaluator(onx, verbose=10)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in example_args])
        )
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @hide_stdout()
    def test_report_results_comparison_ort(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cos", ["X"], ["nx"]),
                    oh.make_node("Sin", ["nx"], ["t"]),
                    oh.make_node("Exp", ["t"], ["u"]),
                    oh.make_node("Log", ["u"], ["uZ"]),
                    oh.make_node("Erf", ["uZ"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        x = torch.rand(5, 6, dtype=torch.float32)
        onnx.checker.check_model(model)
        cmp = ReportResultComparison(dict(r_x=x, r_cos=x.cos(), r_exp=x.cos().sin().exp()))
        cmp.clear()
        feeds = dict(zip([i.name for i in model.graph.input], (x,)))
        rt = OnnxruntimeEvaluator(model, verbose=10)
        rt.run(None, feeds, report_cmp=cmp)
        d = {k: d["abs"] for k, d in cmp.value.items()}
        self.assertLess(d[(0, "nx"), "r_cos"], 1e-6)
        self.assertLess(d[(2, "u"), "r_exp"], 1e-6)

    @hide_stdout()
    def test_skip_layer_normalization(self):
        node = oh.make_node(
            "SkipLayerNormalization",
            ["x", "skip", "beta", "gamma", "bias"],
            ["Z"],
            epsilon=1.0e-5,
            domain="com.microsoft",
        )
        feeds = dict(
            x=self._range(2, 3, 8),
            skip=self._range(2, 3, 8, bias=3),
            beta=self._range(8, bias=1),
            gamma=self._range(8, bias=2),
            bias=self._range(8, bias=0.1),
        )
        ref = ExtendedReferenceEvaluator(node)
        expected = ref.run(None, feeds)
        rt = OnnxruntimeEvaluator(node, verbose=10, opsets={"": 22})
        got = rt.run(None, feeds)
        self.assertEqualAny(expected, got, atol=1e-4)

    @hide_stdout()
    def test_skip_simplified_layer_normalization(self):
        node = oh.make_node(
            "SkipSimplifiedLayerNormalization",
            ["x", "skip", "beta", "gamma"],
            ["Z", "", "", "bias"],
            epsilon=1.0e-5,
            domain="com.microsoft",
        )
        feeds = dict(
            x=self._range(2, 3, 8),
            skip=self._range(2, 3, 8, bias=3),
            beta=self._range(8, bias=1),
            gamma=self._range(8, bias=2),
        )
        rt = OnnxruntimeEvaluator(node, verbose=10, opsets={"": 22})
        got = rt.run(None, feeds)
        self.assertEqual(len(got), 2)
        self.assertIsInstance(got[0], np.ndarray)
        self.assertIsInstance(got[1], np.ndarray)
        self.assertEqual(got[0].shape, feeds["x"].shape)
        self.assertEqual(got[0].dtype, feeds["x"].dtype)
        self.assertEqual(got[1].shape, feeds["x"].shape)
        self.assertEqual(got[1].dtype, feeds["x"].dtype)

    def test_function_proto_with_kwargs(self):
        linear_function = oh.make_function(
            "test_domain",
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("Constant", [], ["eps"]),
                oh.make_node("Constant", [], ["zero"], value_ints=[0]),
                oh.make_node("Unsqueeze", ["eps", "zero"], ["eps1d"]),
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("Add", ["b", "eps1d"], ["beps"]),
                oh.make_node("Add", ["xa", "beps"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            ["epsilon"],
        )
        att = onnx.AttributeProto()
        att.name = "value_float"
        att.ref_attr_name = "epsilon"
        att.type = onnx.AttributeProto.FLOAT
        linear_function.node[0].attribute.append(att)
        feeds = dict(
            x=np.random.rand(4, 4).astype(np.float32),
            a=np.random.rand(4, 2).astype(np.float32),
            b=np.random.rand(1, 2).astype(np.float32),
        )
        epsilon = 15.6
        expected = feeds["x"] @ feeds["a"] + feeds["b"] + epsilon
        sess = OnnxruntimeEvaluator(
            linear_function, whole=True, function_kwargs=dict(epsilon=epsilon)
        )
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @hide_stdout()
    def test_ort_eval_loop_seq(self):
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        _mkv_ = oh.make_tensor_value_info
        model = oh.make_model(
            graph=oh.make_graph(
                name="loop_test",
                inputs=[
                    oh.make_tensor_value_info("trip_count", TINT64, ["a"]),
                    oh.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
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
                                _mkv_("iter_count", TINT64, []),
                                _mkv_("cond_in", onnx.TensorProto.BOOL, []),
                                oh.make_tensor_sequence_value_info("seq_in", TFLOAT, None),
                            ],
                            [
                                _mkv_("cond_out", onnx.TensorProto.BOOL, []),
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
        ev = OnnxruntimeEvaluator(model, verbose=10)
        feeds = dict(trip_count=torch.tensor([3], dtype=torch.int64), cond=torch.tensor(True))
        got = ev.run(None, feeds)
        self.assertEqual((6,), got[0].shape)
        self.assertEqualArray(
            torch.tensor([1.0, 1.0, 2.0, 1.0, 2.0, 3.0], dtype=torch.float32), got[0]
        )
        self.assertIsInstance(got[0], torch.Tensor)

    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def _get_model(self) -> onnx.ModelProto:
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
                [
                    oh.make_tensor_value_info("X", TFLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TFLOAT, [3, 5, 128, 64]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        return model

    @ignore_warnings(DeprecationWarning)
    def test_ort_eval(self):
        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected, out, _ = self.capture(lambda: ref.run(None, feeds)[0])
        self.assertIn("Reshape(xm, shape3) -> Z", out)

        ort_eval = OnnxruntimeEvaluator(model, verbose=10, opsets=20)
        got, out, _ = self.capture(lambda: ort_eval.run(None, feeds)[0])
        self.assertEqualArray(expected, got, atol=1e-4)
        self.assertIn("Reshape(xm, shape3) -> Z", out)

    @ignore_warnings(DeprecationWarning)
    def test__inference(self):
        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        ort_eval = _InferenceSession(model)
        got = ort_eval.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

    @ignore_warnings(DeprecationWarning)
    @requires_cuda()
    @hide_stdout()
    def test_ort_eval_cuda(self):
        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected = ref.run(None, feeds)[0]

        ort_eval = OnnxruntimeEvaluator(model, verbose=10, opsets=20, providers="cuda")
        got = ort_eval.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-1)

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_ort_eval_node_proto(self):
        model = self._get_model()

        feeds = {"X": self._range(32, 128), "zero": np.array([0], dtype=np.int64)}
        ref = ExtendedReferenceEvaluator(model.graph.node[0], verbose=10)
        expected = ref.run(None, feeds)

        ort_eval = OnnxruntimeEvaluator(model.graph.node[0], verbose=10, opsets=20)
        got = ort_eval.run(None, feeds)
        self.assertEqualArrayAny(expected, got, atol=1e-4)
        self.assertIsInstance(expected[0], np.ndarray)
        self.assertIsInstance(got[0], np.ndarray)

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_ort_eval_node_proto_torch(self):
        model = self._get_model()

        feeds_np = {"X": self._range(32, 128), "zero": np.array([0], dtype=np.int64)}
        feeds = {k: torch.from_numpy(v) for k, v in feeds_np.items()}
        ref = ExtendedReferenceEvaluator(model.graph.node[0], verbose=10)
        expected = ref.run(None, feeds_np)

        ort_eval = OnnxruntimeEvaluator(model.graph.node[0], verbose=10, opsets=20)
        got = ort_eval.run(None, feeds)
        self.assertIsInstance(got[0], torch.Tensor)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)

    @hide_stdout()
    def test_local_function(self):
        new_domain = "custom"

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("Add", ["xa", "b"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain),
                oh.make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                oh.make_tensor_value_info("A", TFLOAT, [None, None]),
                oh.make_tensor_value_info("B", TFLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TFLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
            ir_version=10,
        )
        feeds = {
            "X": np.random.randn(3, 3).astype(np.float32),
            "A": np.random.randn(3, 3).astype(np.float32),
            "B": np.random.randn(3, 3).astype(np.float32),
        }
        ref = ExtendedReferenceEvaluator(onnx_model)
        ort_eval = OnnxruntimeEvaluator(onnx_model, verbose=10, opsets=20)
        expected = ref.run(None, feeds)
        got = ort_eval.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    @classmethod
    def _trange(cls, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return torch.from_numpy(x.reshape(tuple(shape)).astype(np.float32))

    @classmethod
    def _get_model_init(cls, itype) -> Tuple[onnx.ModelProto, Dict[str, Any], Tuple[Any, ...]]:
        dtype = tensor_dtype_to_np_dtype(itype)
        ttype = onnx_dtype_to_torch_dtype(itype)
        cst = np.arange(6).astype(dtype)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("IsNaN", ["x"], ["xi"]),
                    oh.make_node("IsNaN", ["y"], ["yi"]),
                    oh.make_node("Cast", ["xi"], ["xii"], to=onnx.TensorProto.INT64),
                    oh.make_node("Cast", ["yi"], ["yii"], to=onnx.TensorProto.INT64),
                    oh.make_node("Add", ["xii", "yii"], ["gggg"]),
                    oh.make_node("Cast", ["gggg"], ["final"], to=itype),
                ],
                "dummy",
                [oh.make_tensor_value_info("x", itype, [None, None])],
                [oh.make_tensor_value_info("final", itype, [None, None])],
                [onh.from_array(cst, name="y")],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
            ir_version=10,
        )
        feeds = {"x": cls._trange(5, 6).to(ttype)}
        expected = torch.isnan(feeds["x"]).to(int) + torch.isnan(
            torch.from_numpy(cst.astype(float))
        ).to(int)
        return (model, feeds, (expected.to(ttype),))

    @hide_stdout()
    def test_init_numpy_afloat32(self):
        model, feeds, expected = self._get_model_init(TFLOAT)
        wrap = OnnxruntimeEvaluator(
            model, providers="cpu", graph_optimization_level=False, verbose=10
        )
        got = wrap.run(None, {k: v.numpy() for k, v in feeds.items()})
        self.assertIsInstance(got[0], np.ndarray)
        self.assertEqualArray(expected[0], got[0])

    @hide_stdout()
    def test_init_numpy_bfloat16(self):
        model, feeds, expected = self._get_model_init(onnx.TensorProto.BFLOAT16)
        wrap = OnnxruntimeEvaluator(
            model, providers="cpu", graph_optimization_level=False, verbose=10
        )
        got = wrap.run(
            None, {k: v.to(float).numpy().astype(ml_dtypes.bfloat16) for k, v in feeds.items()}
        )
        self.assertIsInstance(got[0], np.ndarray)
        self.assertEqualArray(expected[0], got[0])

    def test_init_numpy_bfloat16_whole(self):
        model, feeds, expected = self._get_model_init(onnx.TensorProto.BFLOAT16)
        wrap = OnnxruntimeEvaluator(model, providers="cpu", whole=True)
        got = wrap.run(
            None, {k: v.to(float).numpy().astype(ml_dtypes.bfloat16) for k, v in feeds.items()}
        )
        self.assertIsInstance(got[0], np.ndarray)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqual(got[0].dtype, ml_dtypes.bfloat16)

    @hide_stdout()
    def test_init_torch_afloat32(self):
        model, feeds, expected = self._get_model_init(TFLOAT)
        wrap = OnnxruntimeEvaluator(
            model, providers="cpu", graph_optimization_level=False, verbose=10
        )
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], (torch.Tensor, np.ndarray))
        self.assertEqualArray(expected[0], got[0])

    @hide_stdout()
    def test_init_torch_bfloat16(self):
        model, feeds, expected = self._get_model_init(onnx.TensorProto.BFLOAT16)
        wrap = OnnxruntimeEvaluator(
            model, providers="cpu", graph_optimization_level=False, verbose=10
        )
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], (torch.Tensor, np.ndarray))
        self.assertEqualArray(expected[0], got[0])

    @hide_stdout()
    def test_if(self):
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
                    oh.make_node(
                        "If",
                        ["cond"],
                        ["Z_c"],
                        then_branch=oh.make_graph(
                            [
                                oh.make_node("Constant", [], ["two"], value_floats=[2.1]),
                                oh.make_node("Add", ["X00", "two"], ["Y"]),
                            ],
                            "then",
                            [],
                            [_mkv_("Y")],
                        ),
                        else_branch=oh.make_graph(
                            [
                                oh.make_node("Constant", [], ["two"], value_floats=[2.2]),
                                oh.make_node("Sub", ["X0", "two"], ["Y"]),
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
        feeds = {
            "X": np.array([1, 2, 3], dtype=np.float32),
            "one": np.array([1], dtype=np.float32),
        }
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected = ref.run(None, feeds)[0]
        sess = OnnxruntimeEvaluator(model, verbose=10)
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected[0], got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
