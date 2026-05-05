import unittest
import numpy as np
import onnxruntime
from yobx.ext_test_case import ExtTestCase
from yobx.helpers.einsum_helper import (
    decompose_einsum,
    decompose_einsum_2inputs,
    list_decomposed_nodes,
)


class TestEinsumHelper(ExtTestCase):
    def _run(self, model, inputs: dict) -> np.ndarray:
        """Runs an ONNX model with onnxruntime and returns the single output."""
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, inputs)[0]

    def test_decompose_einsum_matmul(self):
        """Tests that ``ij,jk->ik`` decomposes correctly."""
        model = decompose_einsum("ij,jk->ik", (3, 4), (4, 5))
        self.assertIsNotNone(model)
        self.assertGreater(len(model.graph.node), 0)

        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(4, 5).astype(np.float32)
        result = self._run(model, {"X0": a, "X1": b})
        expected = np.einsum("ij,jk->ik", a, b)
        self.assertAlmostEqual(result, expected, atol=1e-5)

    def test_decompose_einsum_batched_matmul(self):
        """Tests that ``bij,bjk->bik`` decomposes correctly."""
        model = decompose_einsum("bij,bjk->bik", (2, 3, 4), (2, 4, 5))
        a = np.random.rand(2, 3, 4).astype(np.float32)
        b = np.random.rand(2, 4, 5).astype(np.float32)
        result = self._run(model, {"X0": a, "X1": b})
        expected = np.einsum("bij,bjk->bik", a, b)
        self.assertAlmostEqual(result, expected, atol=1e-5)

    def test_decompose_einsum_three_operands(self):
        """Tests a three-operand contraction."""
        model = decompose_einsum("bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))
        x0 = np.random.rand(2, 2, 2).astype(np.float32)
        x1 = np.random.rand(2, 2).astype(np.float32)
        x2 = np.random.rand(2, 2, 2).astype(np.float32)
        result = self._run(model, {"X0": x0, "X1": x1, "X2": x2})
        expected = np.einsum("bac,cd,def->ebc", x0, x1, x2)
        self.assertAlmostEqual(result, expected, atol=1e-5)

    def test_decompose_einsum_without_shapes(self):
        """Tests that input shapes are optional."""
        model = decompose_einsum("ij,jk->ik")
        self.assertIsNotNone(model)
        self.assertGreater(len(model.graph.node), 0)

    def test_decompose_einsum_float64(self):
        """Tests decomposition with float64 dtype."""
        model = decompose_einsum("ij,jk->ik", (3, 4), (4, 5), dtype=np.float64)
        import onnx

        self.assertEqual(model.graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.DOUBLE)

    def test_decompose_einsum_input_names(self):
        """Tests that inputs are named X0, X1, …."""
        model = decompose_einsum("ij,jk->ik", (3, 4), (4, 5))
        input_names = [i.name for i in model.graph.input]
        self.assertIn("X0", input_names)
        self.assertIn("X1", input_names)

    def test_decompose_einsum_output_name(self):
        """Tests that the output is named Z."""
        model = decompose_einsum("ij,jk->ik", (3, 4), (4, 5))
        self.assertEqual(model.graph.output[0].name, "Z")

    def test_list_decomposed_nodes(self):
        """Tests list_decomposed_nodes returns a non-empty list of strings."""
        ops = list_decomposed_nodes("ij,jk->ik")
        self.assertIsInstance(ops, list)
        self.assertGreater(len(ops), 0)
        for op in ops:
            self.assertIsInstance(op, str)

    def test_decompose_einsum_missing_arrow_raises(self):
        """Tests that an equation without ``->`` raises an error."""
        with self.assertRaises((NotImplementedError, AssertionError, ValueError)):
            decompose_einsum("ij,jk")

    def test_decompose_einsum_symbolic_shapes(self):
        """Tests that symbolic (string) dim names appear in the produced ONNX model."""
        model = decompose_einsum("bij,bjk->bik", ("batch", 3, 4), ("batch", 4, 5))
        # Input 0 should have "batch" as a named symbolic dimension.
        dim0 = model.graph.input[0].type.tensor_type.shape.dim[0]
        self.assertEqual(dim0.dim_param, "batch")
        # The model should still run with any concrete batch size.
        a = np.random.rand(2, 3, 4).astype(np.float32)
        b = np.random.rand(2, 4, 5).astype(np.float32)
        result = self._run(model, {"X0": a, "X1": b})
        expected = np.einsum("bij,bjk->bik", a, b)
        self.assertAlmostEqual(result, expected, atol=1e-5)

    def test_decompose_einsum_none_shapes(self):
        """Tests that None dimension values produce fully dynamic ONNX shapes."""
        model = decompose_einsum("ij,jk->ik", (None, 4), (4, None))
        # None dims should be dynamic (dim_param="", dim_value=0 in ONNX).
        dim0_input0 = model.graph.input[0].type.tensor_type.shape.dim[0]
        self.assertFalse(dim0_input0.HasField("dim_value") and dim0_input0.dim_value > 0)
        # The model should still run.
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(4, 5).astype(np.float32)
        result = self._run(model, {"X0": a, "X1": b})
        expected = np.einsum("ij,jk->ik", a, b)
        self.assertAlmostEqual(result, expected, atol=1e-5)
        model = decompose_einsum("ij,jk->ik", (2, 3), (3, 4))
        self.assertGreater(len(model.graph.node), 0)


class TestDecomposeEinsum2Inputs(ExtTestCase):
    """Tests for the new independent 2-input einsum ONNX decomposition algorithm."""

    def _run(self, model, inputs: dict) -> np.ndarray:
        """Runs an ONNX model with onnxruntime and returns the single output."""
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, inputs)[0]

    def _check(self, equation, sh0, sh1, dtype=np.float32):
        """Helper that builds, runs, and validates a 2-input decomposition."""
        a = np.random.rand(*sh0).astype(dtype)
        b = np.random.rand(*sh1).astype(dtype)
        model = decompose_einsum_2inputs(equation, sh0, sh1, dtype=dtype)
        result = self._run(model, {"X0": a, "X1": b})
        expected = np.einsum(equation, a, b)
        self.assertAlmostEqual(result, expected, atol=1e-5)

    # ------------------------------------------------------------------
    # Basic 2-D contractions
    # ------------------------------------------------------------------

    def test_matmul(self):
        """Standard matrix multiplication ``ij,jk->ik``."""
        self._check("ij,jk->ik", (3, 4), (4, 5))

    def test_transposed_matmul(self):
        """Matrix multiplication with transposed second operand ``ij,kj->ik``."""
        self._check("ij,kj->ik", (3, 4), (5, 4))

    def test_matrix_vector(self):
        """Matrix-vector product ``ij,j->i``."""
        self._check("ij,j->i", (3, 4), (4,))

    def test_outer_product(self):
        """Outer product ``i,j->ij`` (no contraction axis)."""
        self._check("i,j->ij", (3,), (5,))

    def test_elementwise_dot(self):
        """Row-wise dot product ``ij,ij->i``."""
        self._check("ij,ij->i", (3, 4), (3, 4))

    def test_dot_to_scalar(self):
        """Full dot product to scalar ``i,i->``."""
        self._check("i,i->", (4,), (4,))

    def test_full_sum_to_scalar(self):
        """Element-wise product summed to scalar ``ij,ij->``."""
        self._check("ij,ij->", (3, 4), (3, 4))

    # ------------------------------------------------------------------
    # Batched contractions
    # ------------------------------------------------------------------

    def test_batched_matmul(self):
        self._check("bij,bjk->bik", (2, 3, 4), (2, 4, 5))

    def test_multi_batch_matmul(self):
        self._check("bcij,bcjk->bcik", (2, 3, 4, 5), (2, 3, 5, 6))

    def test_multi_batch_matmul_4d(self):
        """Multi-batch 4D matmul ``abij,abjk->abik`` (label: multi-batch matmul 4D).

        sym0=('A', 'B', 'I', 'K'), sym1=('A', 'B', 'K', 'N'),
        sh0=(2, 3, 16, 32), sh1=(2, 3, 32, 8).
        """
        self._check("abij,abjk->abik", (2, 3, 16, 32), (2, 3, 32, 8))

    def test_decompose_einsum_symbolic(self):
        decompose_einsum("abij,abjk->abik", ("A", "B", "I", "K"), ("A", "B", "K", "N"))

    def test_decompose_einsum_symbolic_2(self):
        decompose_einsum_2inputs("abij,abjk->abik", ("A", "B", "I", "K"), ("A", "B", "K", "N"))

    def test_multi_batch_matmul_4d_cost_inference(self):
        """Cost inference for ``abij,abjk->abik`` with symbolic dims
        ('A', 'B', 'I', 'K') x ('A', 'B', 'K', 'N') and concrete feeds
        (2, 3, 16, 32) x (2, 3, 32, 8)."""
        from yobx.xshape import BasicShapeBuilder, InferenceMode

        model = decompose_einsum_2inputs(
            "abij,abjk->abik", ("A", "B", "I", "K"), ("A", "B", "K", "N")
        )
        feeds = {
            "X0": np.ones((2, 3, 16, 32), dtype=np.float32),
            "X1": np.ones((2, 3, 32, 8), dtype=np.float32),
        }
        builder = BasicShapeBuilder()
        cost_sym = builder.run_model(model, inference=InferenceMode.COST)
        cost_conc = builder.evaluate_cost_with_true_inputs(feeds, cost_sym)
        total = sum(f or 0 for _, f, _ in cost_conc)
        self.assertGreater(total, 0)
    # ------------------------------------------------------------------
    # Higher-rank contractions
    # ------------------------------------------------------------------

    def test_multi_dim_contraction(self):
        """Higher-rank contraction ``abc,cde->abde``."""
        self._check("abc,cde->abde", (2, 3, 4), (4, 5, 6))

    def test_reordered_output(self):
        """Contraction with non-trivial output permutation ``abc,cd->bad``."""
        self._check("abc,cd->bad", (2, 3, 4), (4, 5))

    def test_batched_with_free_dims(self):
        """Batched contraction with separate free dims ``bik,bkj->bij``."""
        self._check("bik,bkj->bij", (2, 3, 4), (2, 4, 5))

    # ------------------------------------------------------------------
    # Input with shared batch + non-trivial permutation
    # ------------------------------------------------------------------

    def test_transposed_subscripts(self):
        """Subscripts in non-canonical order ``ba,bc->ac``."""
        self._check("ba,bc->ac", (3, 2), (3, 4))

    # ------------------------------------------------------------------
    # dtype support
    # ------------------------------------------------------------------

    def test_float64(self):
        """Decomposition should honour float64 dtype."""
        import onnx

        model = decompose_einsum_2inputs("ij,jk->ik", (3, 4), (4, 5), dtype=np.float64)
        self.assertEqual(model.graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.DOUBLE)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_wrong_number_of_inputs_raises(self):
        """Three-input equation must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            decompose_einsum_2inputs("ij,jk,kl->il")

    def test_missing_arrow_raises(self):
        """Equation without ``->`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            decompose_einsum_2inputs("ij,jk")

    def test_invalid_output_letter_raises(self):
        """Letter in output that is absent from both inputs must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            decompose_einsum_2inputs("ij,jk->iz")

    # ------------------------------------------------------------------
    # Output metadata
    # ------------------------------------------------------------------

    def test_default_input_output_names(self):
        """Default names are X0, X1, Z."""
        model = decompose_einsum_2inputs("ij,jk->ik", (3, 4), (4, 5))
        self.assertEqual(model.graph.input[0].name, "X0")
        self.assertEqual(model.graph.input[1].name, "X1")
        self.assertEqual(model.graph.output[0].name, "Z")

    def test_nodes_use_only_basic_ops(self):
        """Decomposed graph must not contain an Einsum node."""
        model = decompose_einsum_2inputs("ij,jk->ik", (3, 4), (4, 5))
        op_types = {n.op_type for n in model.graph.node}
        self.assertNotIn("Einsum", op_types)
        self.assertIn("MatMul", op_types)

    # ------------------------------------------------------------------
    # Cost inference for 4-D multi-batch equations
    # ------------------------------------------------------------------

    def test_cost_inference_4d_multi_batch_decompose_einsum(self):
        """BasicShapeBuilder.run_model with InferenceMode.COST must succeed for
        a 4-D multi-batch equation using decompose_einsum (strategy A)."""
        from yobx.xshape import BasicShapeBuilder, InferenceMode

        model = decompose_einsum("abij,abjk->abik", (2, 3, 8, 4), (2, 3, 4, 6))
        feeds = {
            "X0": np.ones((2, 3, 8, 4), dtype=np.float32),
            "X1": np.ones((2, 3, 4, 6), dtype=np.float32),
        }
        builder = BasicShapeBuilder()
        cost_sym = builder.run_model(model, inference=InferenceMode.COST)
        cost_conc = builder.evaluate_cost_with_true_inputs(feeds, cost_sym)
        total = sum(f or 0 for _, f, _ in cost_conc)
        self.assertGreater(total, 0)

    def test_cost_inference_4d_multi_batch_decompose_einsum_2inputs(self):
        """BasicShapeBuilder.run_model with InferenceMode.COST must succeed for
        a 4-D multi-batch equation using decompose_einsum_2inputs (strategy B).

        Symbolic (string) dims are used so the shape builder can propagate
        expressions through the graph; concrete values are substituted at
        evaluation time via ``evaluate_cost_with_true_inputs``."""
        from yobx.xshape import BasicShapeBuilder, InferenceMode

        model = decompose_einsum_2inputs(
            "abij,abjk->abik", ("A", "B", "I", "K"), ("A", "B", "K", "L")
        )
        feeds = {
            "X0": np.ones((2, 3, 8, 4), dtype=np.float32),
            "X1": np.ones((2, 3, 4, 6), dtype=np.float32),
        }
        builder = BasicShapeBuilder()
        cost_sym = builder.run_model(model, inference=InferenceMode.COST)
        cost_conc = builder.evaluate_cost_with_true_inputs(feeds, cost_sym)
        total = sum(f or 0 for _, f, _ in cost_conc)
        self.assertGreater(total, 0)

    def test_cost_inference_4d_reduction(self):
        """BasicShapeBuilder.run_model with InferenceMode.COST must succeed for
        a 4-D reduction equation using decompose_einsum_2inputs (strategy B).

        Symbolic (string) dims are used so the shape builder can propagate
        expressions through the graph; concrete values are substituted at
        evaluation time via ``evaluate_cost_with_true_inputs``."""
        from yobx.xshape import BasicShapeBuilder, InferenceMode

        model = decompose_einsum_2inputs("abij,ij->ab", ("A", "B", "I", "J"), ("I", "J"))
        feeds = {
            "X0": np.ones((2, 3, 8, 4), dtype=np.float32),
            "X1": np.ones((8, 4), dtype=np.float32),
        }
        builder = BasicShapeBuilder()
        cost_sym = builder.run_model(model, inference=InferenceMode.COST)
        cost_conc = builder.evaluate_cost_with_true_inputs(feeds, cost_sym)
        total = sum(f or 0 for _, f, _ in cost_conc)
        self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
