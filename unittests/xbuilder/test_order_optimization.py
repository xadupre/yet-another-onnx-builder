import unittest
from typing import Optional
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx.checker import check_model
from yobx.reference import ExtendedReferenceEvaluator
from yobx.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings
from yobx.xbuilder.graph_builder import GraphBuilder, OptimizationOptions
from yobx.xbuilder import OrderAlgorithm

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16


class TestGraphOrderOptimization(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def _fix_shape(self, onx):
        skip_names = set()
        for node in onx.graph.node:
            if node.op_type in {"SequenceConstruct", "SequenceAt"}:
                skip_names |= set(node.output)

        new_shapes = []
        for sh in onx.graph.value_info:
            if sh.name in skip_names:
                continue
            if sh.type.tensor_type.elem_type != 0:
                new_shapes.append(sh)
        del onx.graph.value_info[:]
        onx.graph.value_info.extend(new_shapes)

    def _check_ort_cpu_or_cuda(self, onx):
        def cl(text):
            return (
                text.replace("\n", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
            )

        def s(cond):
            if not cond:
                with open("dump_bug_test_pattern_combination_false.onnx", "wb") as f:
                    f.write(onx.SerializeToString())
            return cond

        for i in onx.graph.input:
            assert s(i.type.tensor_type.elem_type != 0), f"Input {i.name!r} has no type"
        for i in onx.graph.output:
            assert s(i.type.tensor_type.elem_type != 0), f"Output {i.name!r} has no type"

        skip_names = set()
        for node in onx.graph.node:
            if node.op_type in {"SequenceConstruct", "SequenceAt"}:
                skip_names |= set(node.output)

        for sh in onx.graph.value_info:
            if sh.name in skip_names:
                continue
            assert s(sh.type.tensor_type.elem_type != 0), f"Result {sh.name!r} has no type"

        import onnxruntime
        from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument

        opsets = {d.domain: d.version for d in onx.opset_import}
        options = onnxruntime.SessionOptions()
        providers = ["CPUExecutionProvider"]
        if "yaourt.ortops.fused_kernel.cuda" in opsets:
            from yaourt.ortops.fused_kernel.cuda import get_ort_ext_libs

            options.register_custom_ops_library(get_ort_ext_libs()[0])
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "yaourt.ortops.sparse.cuda" in opsets:
            from yaourt.ortops.sparse.cuda import get_ort_ext_libs

            options.register_custom_ops_library(get_ort_ext_libs()[0])

        try:
            onnxruntime.InferenceSession(onx.SerializeToString(), options, providers=providers)
        except (Fail, InvalidArgument) as e:
            if "com.microsoft:SoftmaxGrad(-1) is not a registered function/op" in str(e):
                raise unittest.SkipTest("onnxruntime-training is not installed")
            err = []
            rows = []
            for i in onx.graph.input:
                rows.append(f"input-: {i.name!r} {cl(str(i.type))}")
                if i.type.tensor_type.elem_type == 0:
                    err.append(f"ERR:input-: {i.name!r} {cl(str(i.type))}")
            for i in onx.graph.output:
                rows.append(f"output: {i.name!r} {cl(str(i.type))}")
                if i.type.tensor_type.elem_type == 0:
                    err.append(f"ERR:output: {i.name!r} {cl(str(i.type))}")
            for i in onx.graph.value_info:
                rows.append(f"shape-: {i.name!r} {cl(str(i.type))}")
                if i.type.tensor_type.elem_type == 0:
                    err.append(f"ERR:shape-: {i.name!r} {cl(str(i.type))}")
            msg = "\n".join(err + rows)

            with open("dump_bug_test_pattern_combination.onnx", "wb") as f:
                f.write(onx.SerializeToString())
            raise AssertionError(msg) from e

    @ignore_warnings(RuntimeWarning)
    @hide_stdout()
    def test_arandom_order(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy1"]),
                    oh.make_node("Mul", ["X", "Y"], ["xy2"]),
                    oh.make_node("Sub", ["X", "Y"], ["xy3"]),
                    oh.make_node("Div", ["X", "Y"], ["xy4"]),
                    oh.make_node("Add", ["xy1", "xy2"], ["xy10"]),
                    oh.make_node("Mul", ["xy1", "xy2"], ["xy12"]),
                    oh.make_node("Sub", ["xy3", "xy4"], ["xy13"]),
                    oh.make_node("Div", ["xy3", "xy4"], ["xy14"]),
                    oh.make_node("Add", ["xy10", "xy12"], ["r20"]),
                    oh.make_node("Add", ["xy13", "xy14"], ["r21"]),
                    oh.make_node("Add", ["r21", "r20"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [4, 4]),
                    oh.make_tensor_value_info("Y", TFLOAT, [4, 4]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [4, 4])],
            )
        )
        check_model(model)
        op_types = [n.op_type for n in model.graph.node]

        verbose = 10
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=None, verbose=verbose, order=OrderAlgorithm.RANDOM
            ),
            verbose=0,
        )

        feeds = {"X": self._range(4, 4), "Y": self._range(4, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        opt_onx = gr.to_onnx(optimize=True)
        new_op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertEqual(len(op_types), len(new_op_types))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @hide_stdout()
    def test_order_bigger_model(self):
        _mkv_ = oh.make_tensor_value_info
        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["ids_weight"], ["shape"], start=0, end=2),
                    oh.make_node("Concat", ["shape", "init328"], ["new_shape"], axis=0),
                    oh.make_node("MatMul", ["ids_weight", "A"], ["A1"]),
                    oh.make_node("MatMul", ["ids_weight", "B"], ["B1"]),
                    oh.make_node("MatMul", ["ids_weight", "C"], ["C1"]),
                    oh.make_node("Reshape", ["A1", "new_shape"], ["Areshaped"]),
                    oh.make_node("Reshape", ["B1", "new_shape"], ["Breshaped"]),
                    oh.make_node("Reshape", ["C1", "new_shape"], ["Creshaped"]),
                    oh.make_node("Transpose", ["Areshaped"], ["At"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["Breshaped"], ["Bt"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["Creshaped"], ["Ct"], perm=[0, 2, 1, 3]),
                ],
                "dummy",
                [_mkv_("ids_weight", TFLOAT, ["batch", "seq", 256])],
                [
                    _mkv_("At", TFLOAT, ["batch", 32, "seq", 8]),
                    _mkv_("Bt", TFLOAT, ["batch", 32, "seq", 8]),
                    _mkv_("Ct", TFLOAT, ["batch", 32, "seq", 8]),
                ],
                [
                    onh.from_array(np.array([32, 8], dtype=np.int64), name="init328"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="A"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="B"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="C"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )

        for onx in [model1]:
            self._fix_shape(onx)
            self._check_ort_cpu_or_cuda(onx)

            gr = GraphBuilder(
                onx,
                infer_shapes_options=False,
                optimization_options=OptimizationOptions(
                    patterns=None, verbose=10, order=OrderAlgorithm.RANDOM
                ),
                verbose=0,
            )
            onx = gr.to_onnx(optimize=True)
            self._check_ort_cpu_or_cuda(onx)

    @ignore_warnings(RuntimeWarning)
    @hide_stdout()
    def test_shape_order(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy1"]),
                    oh.make_node("Mul", ["X", "Y"], ["xy2"]),
                    oh.make_node("Sub", ["X", "Y"], ["xy3"]),
                    oh.make_node("Div", ["X", "Y"], ["xy4"]),
                    oh.make_node("Add", ["xy1", "xy2"], ["xy10"]),
                    oh.make_node("Mul", ["xy1", "xy2"], ["xy12"]),
                    oh.make_node("Sub", ["xy3", "xy4"], ["xy13"]),
                    oh.make_node("Div", ["xy3", "xy4"], ["xy14"]),
                    oh.make_node("Add", ["xy10", "xy12"], ["r20"]),
                    oh.make_node("Add", ["xy13", "xy14"], ["r21"]),
                    oh.make_node("Shape", ["xy1"], ["shape"]),
                    oh.make_node("Add", ["r21", "r20"], ["zs"]),
                    oh.make_node("Reshape", ["zs", "shape"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [4, 4]),
                    oh.make_tensor_value_info("Y", TFLOAT, [4, 4]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [4, 4])],
            )
        )
        check_model(model)
        op_types = [n.op_type for n in model.graph.node]

        verbose = 10
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=None, verbose=verbose, order=OrderAlgorithm.SHAPE, passes=("order",)
            ),
            verbose=0,
        )

        feeds = {"X": self._range(4, 4), "Y": self._range(4, 4)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        opt_onx = gr.to_onnx(optimize=True)
        new_op_types = [n.op_type for n in opt_onx.graph.node]
        self.assertEqual(
            [
                "Add",
                "Shape",
                "Mul",
                "Sub",
                "Div",
                "Add",
                "Mul",
                "Sub",
                "Div",
                "Add",
                "Add",
                "Add",
                "Reshape",
            ],
            [n.op_type for n in opt_onx.graph.node],
        )
        self.assertEqual(len(op_types), len(new_op_types))

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
