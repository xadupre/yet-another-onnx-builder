"""
Unit tests for yobx.sklearn.neural_network.bernoulli_rbm converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnBernoulliRBM(ExtTestCase):
    def _make_data(self, n_samples=20, n_features=8, dtype=np.float32, seed=0):
        rng = np.random.RandomState(seed)
        return rng.rand(n_samples, n_features).astype(dtype)

    def test_bernoulli_rbm_basic_float32(self):
        from sklearn.neural_network import BernoulliRBM

        X = self._make_data(dtype=np.float32)
        rbm = BernoulliRBM(n_components=4, n_iter=5, random_state=0)
        rbm.fit(X)

        onx = to_onnx(rbm, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertTrue(
            "MatMul" in op_types or "Gemm" in op_types, f"Expected MatMul or Gemm in {op_types}"
        )
        self.assertIn("Sigmoid", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        expected = rbm.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        (ort_result,) = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_bernoulli_rbm_basic_float64(self):
        from sklearn.neural_network import BernoulliRBM

        X = self._make_data(dtype=np.float64)
        rbm = BernoulliRBM(n_components=4, n_iter=5, random_state=0)
        rbm.fit(X)

        onx = to_onnx(rbm, (X,))

        self.assertEqual(onx.graph.input[0].type.tensor_type.elem_type, 11)  # DOUBLE

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        expected = rbm.transform(X).astype(np.float64)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        (ort_result,) = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_result, atol=1e-10)

    def test_bernoulli_rbm_output_shape_float32(self):
        from sklearn.neural_network import BernoulliRBM

        X = self._make_data(n_samples=10, n_features=6, dtype=np.float32)
        n_components = 5
        rbm = BernoulliRBM(n_components=n_components, n_iter=5, random_state=0)
        rbm.fit(X)

        onx = to_onnx(rbm, (X,))

        self.assertEqual(onx.graph.input[0].type.tensor_type.elem_type, 1)  # FLOAT

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        self.assertEqual(result.shape, (10, n_components))

    def test_bernoulli_rbm_output_shape_float64(self):
        from sklearn.neural_network import BernoulliRBM

        X = self._make_data(n_samples=10, n_features=6, dtype=np.float64)
        n_components = 5
        rbm = BernoulliRBM(n_components=n_components, n_iter=5, random_state=0)
        rbm.fit(X)

        onx = to_onnx(rbm, (X,))

        self.assertEqual(onx.graph.input[0].type.tensor_type.elem_type, 11)  # DOUBLE

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        self.assertEqual(result.shape, (10, n_components))

    def test_bernoulli_rbm_output_in_range_float32(self):
        from sklearn.neural_network import BernoulliRBM

        X = self._make_data(dtype=np.float32)
        rbm = BernoulliRBM(n_components=8, n_iter=5, random_state=0)
        rbm.fit(X)

        onx = to_onnx(rbm, (X,))

        self.assertEqual(onx.graph.input[0].type.tensor_type.elem_type, 1)  # FLOAT

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        # Probabilities must be in [0, 1]
        self.assertTrue(np.all(result >= 0.0), "Output contains values < 0")
        self.assertTrue(np.all(result <= 1.0), "Output contains values > 1")

    def test_bernoulli_rbm_output_in_range_float64(self):
        from sklearn.neural_network import BernoulliRBM

        X = self._make_data(dtype=np.float64)
        rbm = BernoulliRBM(n_components=8, n_iter=5, random_state=0)
        rbm.fit(X)

        onx = to_onnx(rbm, (X,))

        self.assertEqual(onx.graph.input[0].type.tensor_type.elem_type, 11)  # DOUBLE

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        # Probabilities must be in [0, 1]
        self.assertTrue(np.all(result >= 0.0), "Output contains values < 0")
        self.assertTrue(np.all(result <= 1.0), "Output contains values > 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
