"""
Unit tests for yobx.sklearn.decomposition.LatentDirichletAllocation converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestLatentDirichletAllocation(ExtTestCase):
    def _make_data(self, n_samples=30, n_features=15, seed=0):
        rng = np.random.default_rng(seed)
        return rng.poisson(10, size=(n_samples, n_features)).astype(np.float32)

    def _check_lda(self, lda, X_train, X_test, atol=1e-4):
        """Fit *lda*, convert to ONNX, compare transform output."""
        from yobx.sklearn import to_onnx

        lda.fit(X_train)
        onx = to_onnx(lda, (X_test,))

        # Reference evaluator
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]

        expected = lda.transform(X_test).astype(X_test.dtype)
        self.assertEqualArray(expected, result, atol=atol)

        # OnnxRuntime
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=atol)

        # Rows should be normalised (sum to 1).
        row_sums = result.sum(axis=1)
        self.assertEqualArray(np.ones_like(row_sums), row_sums, atol=1e-5)

        return onx

    def test_lda_basic(self):
        """Basic LDA conversion with default parameters."""
        from sklearn.decomposition import LatentDirichletAllocation

        X_train = self._make_data(n_samples=50, n_features=15, seed=0)
        X_test = self._make_data(n_samples=10, n_features=15, seed=1)

        lda = LatentDirichletAllocation(
            n_components=5,
            max_iter=5,
            max_doc_update_iter=20,
            random_state=0,
        )
        onx = self._check_lda(lda, X_train, X_test)

        # Verify ONNX graph contains expected ops.
        op_types = {n.op_type for n in onx.graph.node}
        self.assertIn("MatMul", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("Exp", op_types)
        self.assertIn("Log", op_types)

    def test_lda_single_topic(self):
        """Edge case: single topic."""
        from sklearn.decomposition import LatentDirichletAllocation

        X_train = self._make_data(n_samples=40, n_features=10, seed=2)
        X_test = self._make_data(n_samples=5, n_features=10, seed=3)

        lda = LatentDirichletAllocation(
            n_components=1,
            max_iter=3,
            max_doc_update_iter=10,
            random_state=1,
        )
        self._check_lda(lda, X_train, X_test)

    def test_lda_many_topics(self):
        """More topics than features."""
        from sklearn.decomposition import LatentDirichletAllocation

        X_train = self._make_data(n_samples=60, n_features=20, seed=4)
        X_test = self._make_data(n_samples=8, n_features=20, seed=5)

        lda = LatentDirichletAllocation(
            n_components=10,
            max_iter=5,
            max_doc_update_iter=15,
            random_state=2,
        )
        self._check_lda(lda, X_train, X_test)

    def test_lda_float64(self):
        """Conversion with float64 input."""
        from sklearn.decomposition import LatentDirichletAllocation

        rng = np.random.default_rng(6)
        X_train = rng.poisson(8, size=(40, 12)).astype(np.float64)
        X_test = rng.poisson(8, size=(6, 12)).astype(np.float64)

        lda = LatentDirichletAllocation(
            n_components=4,
            max_iter=5,
            max_doc_update_iter=15,
            random_state=3,
        )
        self._check_lda(lda, X_train, X_test, atol=1e-6)

    def test_lda_output_shape(self):
        """Check the output has the expected shape (N, n_components)."""
        from sklearn.decomposition import LatentDirichletAllocation
        from yobx.sklearn import to_onnx

        n_samples, n_features, n_topics = 7, 10, 3
        X_train = self._make_data(n_samples=30, n_features=n_features, seed=7)
        X_test = self._make_data(n_samples=n_samples, n_features=n_features, seed=8)

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=3,
            max_doc_update_iter=10,
            random_state=4,
        )
        lda.fit(X_train)
        onx = to_onnx(lda, (X_test,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        self.assertEqual(result.shape, (n_samples, n_topics))


if __name__ == "__main__":
    unittest.main(verbosity=2)
