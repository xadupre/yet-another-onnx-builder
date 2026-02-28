import unittest
import numpy as np
import onnx
from onnx import TensorProto
from yobx.ext_test_case import ExtTestCase
from yobx.light_api import start, g, OnnxGraph, ProtoType, Var, Vars


class TestLightApi(ExtTestCase):
    # ------------------------------------------------------------------
    # OnnxGraph construction helpers
    # ------------------------------------------------------------------

    def test_start_returns_onnx_graph(self):
        gr = start()
        self.assertIsInstance(gr, OnnxGraph)
        self.assertIsNone(gr.opset)

    def test_start_with_opset(self):
        gr = start(opset=18)
        self.assertEqual(gr.opset, 18)

    def test_start_with_opsets(self):
        gr = start(opsets={"": 18, "com.microsoft": 1})
        self.assertEqual(gr.opset, 18)
        self.assertIn("com.microsoft", gr.opsets)

    def test_g_returns_graph_proto_type(self):
        gr = g()
        self.assertEqual(gr.proto_type, ProtoType.GRAPH)

    def test_repr(self):
        r = repr(start(opset=17))
        self.assertIn("OnnxGraph", r)

    def test_opset_conflict_raises(self):
        with self.assertRaises(ValueError):
            start(opset=17, opsets={"": 18})

    # ------------------------------------------------------------------
    # vin / vout
    # ------------------------------------------------------------------

    def test_vin_returns_var(self):
        x = start().vin("X")
        self.assertIsInstance(x, Var)
        self.assertEqual(x.name, "X")

    def test_vin_duplicate_raises(self):
        gr = start()
        gr.vin("X")
        with self.assertRaises(ValueError):
            gr.vin("X")

    def test_vout_missing_name_raises(self):
        gr = start()
        with self.assertRaises(ValueError):
            gr.make_output("Y")

    # ------------------------------------------------------------------
    # Single-node models
    # ------------------------------------------------------------------

    def test_neg_model(self):
        onx = start().vin("X").Neg().rename("Y").vout().to_onnx()
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertEqual(len(onx.graph.node), 1)
        self.assertEqual(onx.graph.node[0].op_type, "Neg")
        self.assertEqual(onx.graph.input[0].name, "X")
        self.assertEqual(onx.graph.output[0].name, "Y")

    def test_relu_model(self):
        onx = start().vin("X").Relu().rename("Y").vout().to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Relu")

    def test_cast_model(self):
        onx = (
            start()
            .vin("X")
            .Cast(to=TensorProto.FLOAT16)
            .rename("Y")
            .vout(elem_type=TensorProto.FLOAT16)
            .to_onnx()
        )
        self.assertEqual(onx.graph.node[0].op_type, "Cast")

    def test_identity_model(self):
        onx = start().vin("X").Identity().rename("Y").vout().to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Identity")

    def test_sigmoid_model(self):
        onx = start().vin("X").Sigmoid().rename("Y").vout().to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Sigmoid")

    # ------------------------------------------------------------------
    # Two-input models
    # ------------------------------------------------------------------

    def test_add_model(self):
        onx = (
            start()
            .vin("X")
            .vin("Y")
            .bring("X", "Y")
            .Add()
            .rename("Z")
            .vout()
            .to_onnx()
        )
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertEqual(onx.graph.node[0].op_type, "Add")
        self.assertEqual(len(onx.graph.input), 2)

    def test_matmul_model(self):
        onx = (
            start()
            .vin("A")
            .vin("B")
            .bring("A", "B")
            .MatMul()
            .rename("C")
            .vout()
            .to_onnx()
        )
        self.assertEqual(onx.graph.node[0].op_type, "MatMul")

    def test_mul_model(self):
        onx = (
            start()
            .vin("X")
            .vin("Y")
            .bring("X", "Y")
            .Mul()
            .rename("Z")
            .vout()
            .to_onnx()
        )
        self.assertEqual(onx.graph.node[0].op_type, "Mul")

    # ------------------------------------------------------------------
    # Operator overloads on Var
    # ------------------------------------------------------------------

    def test_operator_overload_add(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        (x + y).rename("Z").vout().to_onnx()

    def test_operator_overload_neg(self):
        gr = start()
        x = gr.vin("X")
        (-x).rename("Y").vout().to_onnx()

    def test_operator_overload_sub(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        (x - y).rename("Z").vout().to_onnx()

    def test_operator_overload_mul(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        (x * y).rename("Z").vout().to_onnx()

    # ------------------------------------------------------------------
    # Constant initializer
    # ------------------------------------------------------------------

    def test_cst_via_graph(self):
        gr = start()
        c = gr.cst(np.array([2.0], dtype=np.float32), name="c")
        self.assertIsInstance(c, Var)
        self.assertEqual(c.name, "c")

    def test_cst_duplicate_raises(self):
        gr = start()
        gr.cst(np.array([1.0], dtype=np.float32), name="c")
        with self.assertRaises(ValueError):
            gr.cst(np.array([2.0], dtype=np.float32), name="c")

    def test_model_with_initializer(self):
        gr = start()
        x = gr.vin("X")
        w = gr.cst(np.ones((4,), dtype=np.float32), name="w")
        (x + w).rename("Y").vout().to_onnx()

    # ------------------------------------------------------------------
    # Rename
    # ------------------------------------------------------------------

    def test_rename_duplicate_raises(self):
        gr = start()
        gr.vin("X")
        gr.vin("Y")
        with self.assertRaises(RuntimeError):
            gr.rename("X", "Y")

    def test_rename_missing_raises(self):
        gr = start()
        with self.assertRaises(RuntimeError):
            gr.rename("X", "Y")

    # ------------------------------------------------------------------
    # Vars helpers
    # ------------------------------------------------------------------

    def test_vars_len(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        self.assertEqual(len(vs), 2)

    def test_vars_getitem(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        self.assertIs(vs[0], x)
        self.assertIs(vs[1], y)

    def test_vars_wrong_type_raises(self):
        gr = start()
        with self.assertRaises(TypeError):
            Vars(gr, 42)

    def test_vars_check_nin_raises(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        with self.assertRaises(RuntimeError):
            vs._check_nin(2)

    def test_vars_rename(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        vs.rename("A", "B")
        self.assertEqual(x.name, "A")
        self.assertEqual(y.name, "B")

    def test_vars_rename_mismatch_raises(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        with self.assertRaises(ValueError):
            vs.rename("A")

    # ------------------------------------------------------------------
    # bring / left_bring / right_bring
    # ------------------------------------------------------------------

    def test_bring_single(self):
        gr = start()
        x = gr.vin("X")
        result = x.bring("X")
        self.assertIsInstance(result, Var)

    def test_bring_multiple(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        result = x.bring("X", "Y")
        self.assertIsInstance(result, Vars)
        self.assertEqual(len(result), 2)

    def test_left_bring(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        result = y.left_bring(x)
        self.assertIsInstance(result, Vars)
        self.assertEqual(result[0].name, "X")
        self.assertEqual(result[1].name, "Y")

    def test_right_bring(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        result = x.right_bring(y)
        self.assertIsInstance(result, Vars)
        self.assertEqual(result[0].name, "X")
        self.assertEqual(result[1].name, "Y")

    # ------------------------------------------------------------------
    # Subgraph (GraphProto)
    # ------------------------------------------------------------------

    def test_g_returns_graph_proto(self):
        gr = g()
        x = gr.vin("X")
        x.Neg().rename("Y").vout()
        proto = gr.to_onnx()
        self.assertIsInstance(proto, onnx.GraphProto)

    # ------------------------------------------------------------------
    # Execution with ExtendedReferenceEvaluator
    # ------------------------------------------------------------------

    def test_run_neg(self):
        from yobx.reference import ExtendedReferenceEvaluator

        onx = start().vin("X").Neg().rename("Y").vout().to_onnx()
        ref = ExtendedReferenceEvaluator(onx)
        x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        (y,) = ref.run(None, {"X": x})
        np.testing.assert_array_equal(y, -x)

    def test_run_add(self):
        from yobx.reference import ExtendedReferenceEvaluator

        onx = (
            start()
            .vin("X")
            .vin("Y")
            .bring("X", "Y")
            .Add()
            .rename("Z")
            .vout()
            .to_onnx()
        )
        ref = ExtendedReferenceEvaluator(onx)
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([3.0, 4.0], dtype=np.float32)
        (z,) = ref.run(None, {"X": x, "Y": y})
        np.testing.assert_array_equal(z, x + y)

    def test_run_with_initializer(self):
        from yobx.reference import ExtendedReferenceEvaluator

        gr = start()
        x = gr.vin("X")
        scale = gr.cst(np.array([2.0], dtype=np.float32), name="scale")
        (x * scale).rename("Y").vout().to_onnx()
        onx = gr.to_onnx()
        ref = ExtendedReferenceEvaluator(onx)
        x_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (y,) = ref.run(None, {"X": x_val})
        np.testing.assert_array_equal(y, x_val * 2.0)

    # ------------------------------------------------------------------
    # v() helper
    # ------------------------------------------------------------------

    def test_v_helper(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        retrieved = x.v("Y")
        self.assertEqual(retrieved.name, "Y")

    # ------------------------------------------------------------------
    # reshape shorthand
    # ------------------------------------------------------------------

    def test_reshape(self):
        onx = (
            start()
            .vin("X")
            .reshape((3, 1))
            .rename("Y")
            .vout()
            .to_onnx()
        )
        self.assertEqual(onx.graph.node[0].op_type, "Reshape")

    # ------------------------------------------------------------------
    # Concat / Gemm from OpsVars
    # ------------------------------------------------------------------

    def test_concat(self):
        onx = (
            start()
            .vin("X")
            .vin("Y")
            .bring("X", "Y")
            .Concat(axis=0)
            .rename("Z")
            .vout()
            .to_onnx()
        )
        self.assertEqual(onx.graph.node[0].op_type, "Concat")

    def test_gemm(self):
        onx = (
            start()
            .vin("A")
            .vin("B")
            .vin("C")
            .bring("A", "B", "C")
            .Gemm()
            .rename("Y")
            .vout()
            .to_onnx()
        )
        self.assertEqual(onx.graph.node[0].op_type, "Gemm")

    # ------------------------------------------------------------------
    # Transpose attributes
    # ------------------------------------------------------------------

    def test_transpose_no_perm(self):
        onx = start().vin("X").Transpose().rename("Y").vout().to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Transpose")

    def test_transpose_with_perm(self):
        onx = start().vin("X").Transpose(perm=[1, 0]).rename("Y").vout().to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Transpose")

    # ------------------------------------------------------------------
    # Unique name generation
    # ------------------------------------------------------------------

    def test_unique_name_collision(self):
        gr = start()
        n1 = gr.unique_name("r")
        n2 = gr.unique_name("r")
        self.assertNotEqual(n1, n2)

    # ------------------------------------------------------------------
    # to_onnx() from Var
    # ------------------------------------------------------------------

    def test_to_onnx_from_var(self):
        var = start().vin("X").Neg().rename("Y").vout()
        onx = var.to_onnx()
        self.assertIsInstance(onx, onnx.ModelProto)


if __name__ == "__main__":
    unittest.main(verbosity=2)
