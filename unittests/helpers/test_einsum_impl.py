"""
Unit tests ported from onnx-extended ``_unittests/ut_tools/test_einsum.py``
and ``_unittests/ut_tools/test_einsum_bug.py`` (MIT licence).

The tests verify the internal einsum decomposition logic that lives in
``yobx.helpers._einsum``.  All imports that previously pointed at
``onnx_extended.tools.einsum`` have been rewritten to use the
self-contained sub-package, and ``CReferenceEvaluator`` calls have been
replaced with ``onnxruntime.InferenceSession``.
"""

import io
import itertools
import unittest
from contextlib import redirect_stdout

import numpy
import onnxruntime
from onnxruntime import InferenceSession

from yobx.ext_test_case import ExtTestCase
from yobx.helpers._einsum.einsum_impl import (
    EinsumSubOp,
    analyse_einsum_equation,
    apply_einsum_sequence,
    decompose_einsum_equation,
)
from yobx.helpers._einsum.einsum_impl_ext import (
    numpy_diagonal,
    numpy_extended_dot,
    numpy_extended_dot_python,
)

# ---------------------------------------------------------------------------
# Tests ported from test_einsum.py
# ---------------------------------------------------------------------------


class TestEinsumImpl(ExtTestCase):
    # ------------------------------------------------------------------
    # numpy_diagonal
    # ------------------------------------------------------------------

    def test_numpy_diagonal(self):
        mat = numpy.arange(8).reshape((2, 2, 2))
        diag = numpy_diagonal(mat, 1, [1, 2])
        self.assertEqualArray(diag, numpy.array([[0, 3], [4, 7]]))
        diag = numpy_diagonal(mat, 2, [1, 2])
        self.assertEqualArray(diag, numpy.array([[0, 3], [4, 7]]))

        diag = numpy_diagonal(mat, 0, [0, 1])
        self.assertEqualArray(diag, numpy.array([[0, 1], [6, 7]]))
        diag = numpy_diagonal(mat, 1, [0, 1])
        self.assertEqualArray(diag, numpy.array([[0, 1], [6, 7]]))

        diag = numpy_diagonal(mat, 0, [0, 2])
        self.assertEqualArray(diag, numpy.array([[0, 2], [5, 7]]))
        diag = numpy_diagonal(mat, 2, [0, 2])
        self.assertEqualArray(diag, numpy.array([[0, 2], [5, 7]]).T)

    # ------------------------------------------------------------------
    # numpy_extended_dot — 2-D cases
    # ------------------------------------------------------------------

    def test_numpy_extended_dot_2_a(self):
        m1 = numpy.arange(4).reshape((2, 2)).astype(numpy.float32) + 10
        m2 = m1 + 90

        self.assertRaise(lambda: numpy_extended_dot(m1, m2.T, [0], [1], [2]), AssertionError)
        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[1], left=[0], right=[2])
        exp = m1 @ m2
        self.assertEqualArray(exp, numpy.squeeze(dot))
        dot2 = numpy_extended_dot_python(dm1, dm2, axes=[1], left=[0], right=[2])
        self.assertEqualArray(exp, numpy.squeeze(dot2))

        dm1 = m1.reshape((2, 1, 2))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0], right=[1])
        exp = m1 @ m2.T
        self.assertEqualArray(exp, numpy.squeeze(dot))
        dot2 = numpy_extended_dot_python(dm1, dm2, axes=[2], left=[0], right=[1])
        self.assertEqualArray(exp, numpy.squeeze(dot2))

    def test_numpy_extended_dot_2_b(self):
        m1 = numpy.arange(4).reshape((2, 2)).astype(numpy.float32) + 10
        m2 = m1 + 90
        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0], right=[1, 2])
        dot2 = numpy_extended_dot_python(dm1, dm2, axes=[2], left=[0], right=[1, 2])
        self.assertEqualArray(dot, numpy.squeeze(dot2))

    def test_numpy_extended_dot_2_b2(self):
        m1 = numpy.arange(4).reshape((2, 2)).astype(numpy.float32) + 10
        m2 = m1 + 90
        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0, 1], right=[2])
        dot2 = numpy_extended_dot_python(dm1, dm2, axes=[2], left=[0, 1], right=[2])
        self.assertEqualArray(dot, numpy.squeeze(dot2))

    # ------------------------------------------------------------------
    # numpy_extended_dot — 3-D cases
    # ------------------------------------------------------------------

    def test_numpy_extended_dot_3(self):
        m1 = numpy.arange(8).reshape((2, 2, 2)) + 10
        m2 = m1 + 90

        dot = numpy_extended_dot(m1, m2, [1], [0], [2])
        dot2 = numpy_extended_dot_python(m1, m2, [1], [0], [2])
        self.assertEqualArray(dot, dot2)

        dot = numpy_extended_dot(m1, m2, [1], [2], [0])
        dot2 = numpy_extended_dot_python(m1, m2, [1], [2], [0])
        self.assertEqualArray(dot, dot2)

    def test_numpy_extended_dot_3b(self):
        m1 = numpy.arange(8).reshape((2, 2, 2)) + 10
        m2 = m1 + 90

        dot = numpy_extended_dot(m1, m2, [1], [2], [0, 1])
        dot2 = numpy_extended_dot_python(m1, m2, [1], [2], [0, 1])
        self.assertEqualArray(dot, dot2)

    # ------------------------------------------------------------------
    # analyse_einsum_equation
    # ------------------------------------------------------------------

    def test_analyse_einsum_equation(self):
        self.assertRaise(lambda: analyse_einsum_equation("abc"), NotImplementedError)
        self.assertRaise(lambda: analyse_einsum_equation("abc0,ch->ah"), ValueError)
        self.assertRaise(lambda: analyse_einsum_equation("abc,ch->a0"), AssertionError)
        res = analyse_einsum_equation("abc,ch->ah")
        self.assertEqual(len(res), 4)
        letters, mat, lengths, duplicates = res
        self.assertEqual(letters, "abch")
        self.assertEqual(lengths, [3, 2, 2])
        self.assertEqualArray(
            mat, numpy.array([[0, 1, 2, -1], [-1, -1, 0, 1], [0, -1, -1, 1]], dtype=numpy.int8)
        )
        self.assertEqual(duplicates, [None, None, None])

    def test_analyse_einsum_equation_duplicates(self):
        res = analyse_einsum_equation("aac,ca->aa")
        self.assertEqual(len(res), 4)
        letters, mat, lengths, duplicates = res
        self.assertEqual(letters, "ac")
        self.assertEqual(lengths, [3, 2, 2])
        self.assertEqual(duplicates, [{"a": [0, 1], "c": [2]}, None, {"a": [0, 1]}])
        self.assertEqualArray(mat, numpy.array([[1, 2], [1, 0], [1, -1]], dtype=numpy.int8))

    # ------------------------------------------------------------------
    # decompose_einsum_equation — exception cases
    # ------------------------------------------------------------------

    def test_decompose_einsum_equation_exc(self):
        self.assertRaise(
            lambda: decompose_einsum_equation(
                "abc,ch->ah", (2, 2, 2), (2, 2), strategy="donotexist"
            ),
            ValueError,
        )
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2, 2), (2, 2), "donotexist"),
            TypeError,
        )
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2, 2)), AssertionError
        )
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2), (2, 2)), AssertionError
        )

    # ------------------------------------------------------------------
    # decompose_einsum_equation — core tests
    # ------------------------------------------------------------------

    def test_decompose_einsum_equation(self):
        m1 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2))
        m2 = numpy.arange(0, 4).astype(numpy.float32).reshape((2, 2))
        exp = numpy.einsum("bac,ch->ah", m1, m2)

        def fct():
            print("########################## DECOMPOSE")
            seq = decompose_einsum_equation("bac,ch->ah", (2, 2, 2), (2, 2), verbose=True)
            print("########################## APPLY")
            dot = seq.to_dot()
            print(dot)
            red = dot.split("red")
            self.assertEqual(len(red), 5)
            res = apply_einsum_sequence(seq, m1, m2, verbose=True)
            print("########################## END")
            return res

        f = io.StringIO()
        try:
            with redirect_stdout(f):
                res = fct()
        except Exception as e:
            raise AssertionError(f"Issue. Logs =\n{f.getvalue()}") from e

        out = f.getvalue()
        self.assertIn("numpy_extended_dot", out)
        self.assertEqualArray(exp, res)

    def test_decompose_einsum_equation_mm(self):
        m1 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2))
        m2 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2)) + 10
        exp = numpy.einsum("bac,chg->ah", m1, m2)

        def fct():
            print("########################## DECOMPOSE")
            seq = decompose_einsum_equation(
                "bac,chg->ah", (2, 2, 2), (2, 2, 2), verbose=True, clean=True, strategy="numpy"
            )
            print("########################## APPLY")
            dot = seq.to_dot()
            print(dot)
            red = dot.split("red")
            self.assertEqual(len(red), 6)
            res = apply_einsum_sequence(seq, m1, m2, verbose=True)
            print("########################## END")
            onx = seq.to_onnx("Y", "X1", "X2", verbose=True)
            self.assertNotEmpty(onx)
            return res

        f = io.StringIO()
        try:
            with redirect_stdout(f):
                res = fct()
        except Exception as e:
            raise AssertionError(f"Issue. Logs =\n{f.getvalue()}") from e

        out = f.getvalue()
        self.assertIn("batch_dot", out)
        self.assertEqualArray(exp, res)

    def test_decompose_einsum_equation_py_noshape(self):
        m1 = numpy.arange(0, 24).astype(numpy.float32).reshape((2, 3, 4))
        m2 = numpy.arange(0, 20).astype(numpy.float32).reshape((4, 5))
        verbose = False
        for strat, opname in [("numpy", "batch_dot"), ("simple", "matmul")]:
            with self.subTest(strategy=strat):
                seq = decompose_einsum_equation("bac,ch->ah", strategy=strat, verbose=verbose)
                self.assertIn(opname, seq.to_dot())
                res1 = apply_einsum_sequence(seq, m1, m2, verbose=verbose)
                res2 = apply_einsum_sequence(seq, m1, m2, matmul_impl="py", verbose=verbose)
                if strat == "simple":
                    self.assertRaise(
                        lambda seq=seq: apply_einsum_sequence(seq, m1, m2, matmul_impl="py2"),
                        ValueError,
                    )
                self.assertEqualArray(res1, res2)

    def test_decompose_einsum_equation_py(self):
        m1 = numpy.arange(0, 24).astype(numpy.float32).reshape((2, 3, 4))
        m2 = numpy.arange(0, 20).astype(numpy.float32).reshape((4, 5))
        verbose = False
        for strat, opname in [("numpy", "batch_dot"), ("simple", "matmul")]:
            with self.subTest(strategy=strat):
                seq = decompose_einsum_equation(
                    "bac,ch->ah", (2, 3, 4), (4, 5), strategy=strat, verbose=verbose
                )
                self.assertIn(opname, seq.to_dot())
                res1 = apply_einsum_sequence(seq, m1, m2, verbose=verbose)
                res2 = apply_einsum_sequence(seq, m1, m2, matmul_impl="py", verbose=verbose)
                if strat == "simple":
                    self.assertRaise(
                        lambda seq=seq: apply_einsum_sequence(seq, m1, m2, matmul_impl="py2"),
                        ValueError,
                    )
                self.assertEqualArray(res1, res2)

    def test_decompose_einsum_equation_onnx(self):
        m1 = numpy.arange(0, 24).astype(numpy.float32).reshape((2, 3, 4))
        m2 = numpy.arange(0, 20).astype(numpy.float32).reshape((4, 5))
        verbose = False
        for strat in ["numpy"]:
            with self.subTest(strategy=strat):
                seq = decompose_einsum_equation(
                    "bac,ch->ah", (2, 3, 4), (4, 5), strategy=strat, verbose=verbose
                )
                res1 = apply_einsum_sequence(seq, m1, m2, verbose=verbose)
                self.assertRaise(
                    lambda seq=seq: seq.to_onnx("Y", "X1", "X2", dtype=numpy.float32),
                    NotImplementedError,
                )
                seq.simplify_mm_nodes()
                seq.clean_unused_nodes()
                onx = seq.to_onnx("Y", "X1", "X2", dtype=numpy.float32)

                oinf = InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                (res2,) = oinf.run(
                    None, {"X1": m1.astype(numpy.float32), "X2": m2.astype(numpy.float32)}
                )
                self.assertEqualArray(res1, res2)

    def test_decompose_einsum_equation_onnx2(self):
        m1 = numpy.arange(0, 24).astype(numpy.float32).reshape((2, 3, 4))
        m2 = numpy.arange(0, 20).astype(numpy.float32).reshape((4, 5))
        m3 = numpy.arange(0, 77 * 5).astype(numpy.float32).reshape((5, 7, 11))
        verbose = False
        for strat in ["numpy"]:
            with self.subTest(strategy=strat):
                seq = decompose_einsum_equation(
                    "bac,cd,def->ebc",
                    (2, 3, 4),
                    (4, 5),
                    (5, 7, 11),
                    strategy=strat,
                    verbose=verbose,
                )
                res1 = apply_einsum_sequence(seq, m1, m2, m3, verbose=verbose)
                seq.simplify_mm_nodes()
                seq.clean_unused_nodes()
                onx = seq.to_onnx("Y", "X1", "X2", "X3", dtype=numpy.float32)

                oinf = InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                (res2,) = oinf.run(
                    None,
                    {
                        "X1": m1.astype(numpy.float32),
                        "X2": m2.astype(numpy.float32),
                        "X3": m3.astype(numpy.float32),
                    },
                )
                self.assertEqualArray(res1, res2)

                so = onnxruntime.SessionOptions()
                so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                oinf2 = InferenceSession(
                    onx.SerializeToString(), so, providers=["CPUExecutionProvider"]
                )
                (res3,) = oinf2.run(
                    None,
                    {
                        "X1": m1.astype(numpy.float32),
                        "X2": m2.astype(numpy.float32),
                        "X3": m3.astype(numpy.float32),
                    },
                )
                self.assertEqualArray(res1, res3)

    def test_decompose_einsum_equation_pyf(self):
        m1 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2))
        m2 = numpy.arange(0, 4).astype(numpy.float32).reshape((2, 2))
        seq = decompose_einsum_equation("bac,ch->ah", (2, 2, 2), (2, 2))
        res1 = apply_einsum_sequence(seq, m1, m2)
        res2 = apply_einsum_sequence(seq, m1, m2, matmul_impl="pyf")
        self.assertEqualArray(res1, res2)

    # ------------------------------------------------------------------
    # EinsumSubOp construction
    # ------------------------------------------------------------------

    def test_einsum_sub_op(self):
        self.assertRaise(lambda: EinsumSubOp(2, "er", (2, 2)), AssertionError)
        self.assertRaise(lambda: EinsumSubOp(2, "expand_dims"), AssertionError)
        self.assertRaise(lambda: EinsumSubOp(2, "matmul", (2, 2)), RuntimeError)
        self.assertRaise(lambda: EinsumSubOp(2, "id", (2, 2)), AssertionError)

    # ------------------------------------------------------------------
    # Diagonal (repeated-index) equations
    # ------------------------------------------------------------------

    def test_case_1_iii_ii_i(self):
        """Diagonal equation ``ii->i`` via numpy apply path."""
        verbose = False
        equation = "ii->i"
        m1 = numpy.arange(2 * 2).reshape((2, 2)) + 10
        exp = numpy.einsum(equation, m1)
        seq = decompose_einsum_equation(equation, m1.shape, verbose=verbose)
        res = apply_einsum_sequence(seq, m1, verbose=verbose)
        self.assertEqualArray(exp, res)

    def test_case_1_iii_ii_i_j(self):
        """Diagonal equation ``iij->ij`` via numpy apply path."""
        verbose = False
        equation = "iij->ij"
        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        exp = numpy.einsum(equation, m1)
        seq = decompose_einsum_equation(equation, m1.shape, verbose=verbose)
        dot = seq.to_dot()
        self.assertIn("i=0,1", dot)
        res = apply_einsum_sequence(seq, m1, verbose=verbose)
        self.assertEqualArray(exp, res)

    # ------------------------------------------------------------------
    # Two-operand permutation tests
    # ------------------------------------------------------------------

    def _common_test_case_2(self, equation, verbose=False, strategy="simple"):
        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = numpy.arange(4).reshape((2, 2)) + 100
        exp = numpy.einsum(equation, m1, m2)

        seq = decompose_einsum_equation(
            equation, m1.shape, m2.shape, verbose=verbose, strategy=strategy
        )
        res = apply_einsum_sequence(seq, m1, m2, verbose=verbose)
        self.assertEqualArray(exp, res)

    def test_case_2_A(self):
        for strat in ["numpy", "simple"]:
            with self.subTest(strategy=strat):
                self._common_test_case_2("abc,cd->abc", strategy=strat, verbose=False)

    def test_many_2(self):
        """Exhaustive 2-operand equation check (numpy + ONNX via InferenceSession)."""
        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)).astype(numpy.float32) + 10
        m2 = numpy.arange(4).reshape((2, 2)).astype(numpy.float32) + 100

        equations = []
        for p1 in itertools.permutations(list("abc")):
            for p2 in itertools.permutations(list("cd")):
                for i in [1, 2]:
                    for j in [0, 1]:
                        sp1 = "".join(p1)
                        sp2 = "".join(p2)
                        if len({sp1[0], sp1[i], sp2[j]}) != 3:
                            continue
                        equation = f"{sp1},{sp2}->{sp1[0]}{sp1[i]}{sp2[j]}"
                        try:
                            r = numpy.einsum(equation, m1, m2)
                            equations.append((equation, r))
                        except ValueError:
                            continue

        for i, (eq, exp) in enumerate(equations):
            with self.subTest(equation=eq, index=i, total=len(equations)):
                seq = decompose_einsum_equation(eq, m1.shape, m2.shape)
                res = apply_einsum_sequence(seq, m1, m2)
                self.assertEqualArray(exp, res)

                seq2 = decompose_einsum_equation(
                    eq, m1.shape, m2.shape, strategy="numpy", clean=True
                )
                res2 = apply_einsum_sequence(seq2, m1, m2)
                self.assertEqualArray(exp, res2)

                onx = seq2.to_onnx("Y", "X1", "X2", dtype=numpy.float32)
                oinf = InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                (got,) = oinf.run(
                    None, {"X1": m1.astype(numpy.float32), "X2": m2.astype(numpy.float32)}
                )
                self.assertEqualArray(exp, got)

    def test_many_3(self):
        """Exhaustive 3-operand equation check (numpy apply path)."""
        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)).astype(numpy.float32) + 10
        m2 = numpy.arange(4).reshape((2, 2)).astype(numpy.float32) + 100
        m3 = numpy.arange(8).reshape((2, 2, 2)).astype(numpy.float32) + 1000

        equations = []
        for p1 in itertools.permutations(list("abc")):
            for p2 in itertools.permutations(list("cd")):
                for p3 in itertools.permutations(list("def")):
                    for i in [1, 2]:
                        for j in [0, 1]:
                            sp1 = "".join(p1)
                            sp2 = "".join(p2)
                            sp3 = "".join(p3)
                            equation = f"{sp1},{sp2},{sp3}->{sp1[0]}{sp1[i]}{sp3[j]}"
                            try:
                                r = numpy.einsum(equation, m1, m2, m3)
                                equations.append((equation, r))
                            except ValueError:
                                continue

        for i, (eq, exp) in enumerate(equations):
            with self.subTest(equation=eq, index=i, total=len(equations)):
                seq = decompose_einsum_equation(eq, m1.shape, m2.shape, m3.shape)
                res = apply_einsum_sequence(seq, m1, m2, m3)
                self.assertEqualArray(exp, res)


# ---------------------------------------------------------------------------
# Tests ported from test_einsum_bug.py
# ---------------------------------------------------------------------------


class TestEinsumBug(ExtTestCase):
    def test_abbba(self):
        res = decompose_einsum_equation("ab,b->ba", strategy="numpy", clean=True)
        self.assertNotEmpty(res)

    def test__pprint_forward(self):
        res = decompose_einsum_equation("ab,b->ba", strategy="numpy", clean=True)
        pf = res._pprint_forward()
        spl = pf.split("<- id")
        self.assertEqual(len(spl), 4)

    def _common_test_equation_onnx(self, equation, dim1, dim2):
        """Decompose *equation*, run via ONNX, verify numerics."""
        seq = decompose_einsum_equation(equation, clean=True, strategy="numpy")
        onx = seq.to_onnx("Y", "X1", "X2")
        a = numpy.random.rand(*list((2,) * dim1)).astype(numpy.float32)
        b = numpy.random.rand(*list((2,) * dim2)).astype(numpy.float32)
        oinf = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        (got,) = oinf.run(None, {"X1": a, "X2": b})
        expected = numpy.einsum(equation, a, b)
        self.assertEqualArray(expected, got, atol=1e-5)

    def test_decompose_einsum_abc_cde_abde(self):
        self._common_test_equation_onnx("abc,cde->abde", 3, 3)

    def test_decompose_einsum_abcd_cde_abe(self):
        self._common_test_equation_onnx("abcd,cde->abe", 4, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
