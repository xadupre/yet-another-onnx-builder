import unittest
import numpy as np
import onnxruntime
from yobx.ext_test_case import ExtTestCase
from yobx.helpers.einsum_helper import decompose_einsum, list_decomposed_nodes


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
        self.assertGreater(len(model.graph.node), 1)

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
        self.assertGreater(len(model.graph.node), 1)

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
        self.assertGreater(len(model.graph.node), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
