import unittest
import numpy as np
import onnx
from onnx import TensorProto
from yobx.ext_test_case import ExtTestCase
from yobx.builder.light import start, OnnxGraph, Var, Vars


class TestVars(ExtTestCase):
    """Unit tests for the :class:`Vars` class."""

    # ------------------------------------------------------------------
    # __init__ – construction paths
    # ------------------------------------------------------------------

    def test_init_with_var_instances(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        self.assertIsInstance(vs, Vars)
        self.assertEqual(len(vs), 2)

    def test_init_with_string_names(self):
        gr = start()
        gr.vin("X")
        gr.vin("Y")
        vs = Vars(gr, "X", "Y")
        self.assertIsInstance(vs, Vars)
        self.assertEqual(len(vs), 2)
        self.assertEqual(vs[0].name, "X")
        self.assertEqual(vs[1].name, "Y")

    def test_init_with_numpy_array(self):
        gr = start()
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vs = Vars(gr, arr)
        self.assertEqual(len(vs), 1)
        self.assertIsInstance(vs[0], Var)

    def test_init_with_numpy_scalar(self):
        gr = start()
        scalar = np.float32(2.0)
        vs = Vars(gr, scalar)
        self.assertEqual(len(vs), 1)
        self.assertIsInstance(vs[0], Var)

    def test_init_with_python_scalar(self):
        gr = start()
        vs = Vars(gr, 3.14)
        self.assertEqual(len(vs), 1)
        self.assertIsInstance(vs[0], Var)

    def test_init_mixed_var_and_string(self):
        gr = start()
        x = gr.vin("X")
        gr.vin("Y")
        vs = Vars(gr, x, "Y")
        self.assertEqual(len(vs), 2)
        self.assertEqual(vs[0].name, "X")
        self.assertEqual(vs[1].name, "Y")

    def test_init_invalid_type_raises(self):
        gr = start()
        with self.assertRaises(TypeError):
            Vars(gr, object())

    def test_init_wrong_parent_type_raises(self):
        with self.assertRaises(TypeError):
            Vars("not_a_graph")

    # ------------------------------------------------------------------
    # __len__
    # ------------------------------------------------------------------

    def test_len_empty(self):
        gr = start()
        vs = Vars(gr)
        self.assertEqual(len(vs), 0)

    def test_len_one(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        self.assertEqual(len(vs), 1)

    def test_len_three(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        z = gr.vin("Z")
        vs = Vars(gr, x, y, z)
        self.assertEqual(len(vs), 3)

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def test_repr(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        r = repr(vs)
        self.assertIn("Vars", r)
        self.assertIn("X", r)
        self.assertIn("Y", r)

    def test_repr_empty(self):
        gr = start()
        vs = Vars(gr)
        self.assertEqual(repr(vs), "Vars()")

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def test_getitem_first(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        self.assertIs(vs[0], x)

    def test_getitem_last(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        self.assertIs(vs[1], y)

    # ------------------------------------------------------------------
    # _check_nin
    # ------------------------------------------------------------------

    def test_check_nin_passes(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        result = vs._check_nin(2)
        self.assertIs(result, vs)

    def test_check_nin_raises(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        with self.assertRaises(RuntimeError):
            vs._check_nin(2)

    def test_check_nin_zero_raises(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        with self.assertRaises(RuntimeError):
            vs._check_nin(0)

    # ------------------------------------------------------------------
    # rename
    # ------------------------------------------------------------------

    def test_rename_single(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        vs.rename("A")
        self.assertEqual(x.name, "A")

    def test_rename_multiple(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        result = vs.rename("A", "B")
        self.assertIs(result, vs)
        self.assertEqual(x.name, "A")
        self.assertEqual(y.name, "B")

    def test_rename_wrong_count_raises(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        with self.assertRaises(ValueError):
            vs.rename("A")

    def test_rename_too_many_raises(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        with self.assertRaises(ValueError):
            vs.rename("A", "B")

    # ------------------------------------------------------------------
    # vout
    # ------------------------------------------------------------------

    def test_vout_single(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        result = vs.vout()
        self.assertIs(result, vs)
        onx = gr.to_onnx()
        self.assertEqual(len(onx.graph.output), 1)
        self.assertEqual(onx.graph.output[0].name, "X")

    def test_vout_multiple(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        # Build a TopK node that has two outputs
        vs_in = Vars(gr, x, gr.cst(np.array(2, dtype=np.int64), name="k"))
        topk_out = vs_in.TopK(axis=0)
        self.assertIsInstance(topk_out, Vars)
        result = topk_out.vout()
        self.assertIs(result, topk_out)
        onx = gr.to_onnx()
        self.assertEqual(len(onx.graph.output), 2)

    def test_vout_with_elem_type(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        vs.vout(elem_type=TensorProto.FLOAT16)
        onx = gr.to_onnx()
        self.assertEqual(onx.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT16)

    # ------------------------------------------------------------------
    # Inherited BaseVar helpers
    # ------------------------------------------------------------------

    def test_vin_from_vars(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        new_var = vs.vin("Z")
        self.assertIsInstance(new_var, Var)
        self.assertEqual(new_var.name, "Z")

    def test_cst_from_vars(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        c = vs.cst(np.array([2.0], dtype=np.float32), name="c")
        self.assertIsInstance(c, Var)
        self.assertEqual(c.name, "c")

    def test_v_from_vars(self):
        gr = start()
        x = gr.vin("X")
        gr.vin("Y")
        vs = Vars(gr, x)
        retrieved = vs.v("Y")
        self.assertIsInstance(retrieved, Var)
        self.assertEqual(retrieved.name, "Y")

    def test_bring_single_from_vars(self):
        gr = start()
        x = gr.vin("X")
        vs = Vars(gr, x)
        result = vs.bring("X")
        self.assertIsInstance(result, Var)

    def test_bring_multiple_from_vars(self):
        gr = start()
        x = gr.vin("X")
        gr.vin("Y")
        vs = Vars(gr, x)
        result = vs.bring("X", "Y")
        self.assertIsInstance(result, Vars)
        self.assertEqual(len(result), 2)

    def test_left_bring(self):
        # left_bring is designed to be called from a Var: *vars first, then self
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        result = y.left_bring(x)
        self.assertIsInstance(result, Vars)
        self.assertEqual(result[0].name, "X")
        self.assertEqual(result[1].name, "Y")

    def test_right_bring(self):
        # right_bring is designed to be called from a Var: self first, then *vars
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        result = x.right_bring(y)
        self.assertIsInstance(result, Vars)
        self.assertEqual(result[0].name, "X")
        self.assertEqual(result[1].name, "Y")

    def test_to_onnx_from_vars(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        vs[0].Neg().rename("NX").vout()
        onx = vs.to_onnx()
        self.assertIsInstance(onx, onnx.ModelProto)

    def test_make_node_from_vars(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        result = vs.make_node("Add", x, y)
        self.assertIsInstance(result, Var)

    # ------------------------------------------------------------------
    # OpsVars operator methods
    # ------------------------------------------------------------------

    def test_add(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        result = vs.Add()
        result.rename("Z").vout()
        onx = gr.to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Add")

    def test_sub(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        result = vs.Sub()
        result.rename("Z").vout()
        onx = gr.to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Sub")

    def test_mul(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        result = vs.Mul()
        result.rename("Z").vout()
        onx = gr.to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Mul")

    def test_matmul(self):
        gr = start()
        a = gr.vin("A")
        b = gr.vin("B")
        vs = Vars(gr, a, b)
        result = vs.MatMul()
        result.rename("C").vout()
        onx = gr.to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "MatMul")

    def test_concat(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        result = vs.Concat(axis=0)
        result.rename("Z").vout()
        onx = gr.to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Concat")

    def test_gemm(self):
        gr = start()
        a = gr.vin("A")
        b = gr.vin("B")
        c = gr.vin("C")
        vs = Vars(gr, a, b, c)
        result = vs.Gemm()
        result.rename("Y").vout()
        onx = gr.to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Gemm")

    def test_topk_returns_vars(self):
        gr = start()
        x = gr.vin("X")
        k = gr.cst(np.array(2, dtype=np.int64), name="k")
        vs = Vars(gr, x, k)
        result = vs.TopK(axis=0)
        self.assertIsInstance(result, Vars)
        self.assertEqual(len(result), 2)
        result.rename("values", "indices").vout()
        onx = gr.to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "TopK")
        self.assertEqual(len(onx.graph.output), 2)

    def test_reshape(self):
        gr = start()
        x = gr.vin("X")
        shape = gr.cst(np.array([3, 1], dtype=np.int64), name="shape")
        vs = Vars(gr, x, shape)
        result = vs.Reshape()
        result.rename("Y").vout()
        onx = gr.to_onnx()
        self.assertEqual(onx.graph.node[0].op_type, "Reshape")

    # ------------------------------------------------------------------
    # End-to-end ONNX graph construction via Vars
    # ------------------------------------------------------------------

    def test_full_model_add(self):
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
        self.assertEqual(len(onx.graph.output), 1)
        self.assertEqual(onx.graph.output[0].name, "Z")

    def test_full_model_topk(self):
        gr = start()
        x = gr.vin("X")
        k = gr.cst(np.array(3, dtype=np.int64), name="k")
        values, indices = Vars(gr, x, k).TopK(axis=0)
        values.rename("values").vout()
        indices.rename("indices").vout()
        onx = gr.to_onnx()
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertEqual(onx.graph.node[0].op_type, "TopK")
        self.assertEqual(len(onx.graph.output), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
