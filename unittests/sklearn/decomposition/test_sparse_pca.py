"""
Unit tests for yobx.sklearn.decomposition.sparse_pca converter.
Covers SparsePCA and MiniBatchSparsePCA.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestSparsePCA(ExtTestCase):
    def test_sparse_pca_float32(self):
        from sklearn.decomposition import SparsePCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 5)).astype(np.float32)
        sp = SparsePCA(n_components=3, ridge_alpha=0.01, random_state=0)
        sp.fit(X)

        onx = to_onnx(sp, (X,))

        # Check that Sub (centering) and MatMul (projection) are present.
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)

        # Check numerical output.
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = sp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_sparse_pca_float64(self):
        from sklearn.decomposition import SparsePCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 5)).astype(np.float64)
        sp = SparsePCA(n_components=3, ridge_alpha=0.01, random_state=0)
        sp.fit(X)

        onx = to_onnx(sp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = sp.transform(X)
        self.assertEqualArray(expected, result, atol=1e-8)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-8)

    def test_sparse_pca_ridge_alpha(self):
        from sklearn.decomposition import SparsePCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((20, 5)).astype(np.float32)
        sp = SparsePCA(n_components=2, ridge_alpha=0.1, random_state=0)
        sp.fit(X)

        onx = to_onnx(sp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = sp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_mini_batch_sparse_pca_float32(self):
        from sklearn.decomposition import MiniBatchSparsePCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((20, 5)).astype(np.float32)
        mbsp = MiniBatchSparsePCA(n_components=3, ridge_alpha=0.01, random_state=0)
        mbsp.fit(X)

        onx = to_onnx(mbsp, (X,))

        # Check that Sub (centering) and MatMul (projection) are present.
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)

        # Check numerical output.
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mbsp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_mini_batch_sparse_pca_float64(self):
        from sklearn.decomposition import MiniBatchSparsePCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X = rng.standard_normal((20, 5)).astype(np.float64)
        mbsp = MiniBatchSparsePCA(n_components=3, ridge_alpha=0.01, random_state=0)
        mbsp.fit(X)

        onx = to_onnx(mbsp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = mbsp.transform(X)
        self.assertEqualArray(expected, result, atol=1e-8)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-8)

    def test_pipeline_sparse_pca_logistic_regression(self):
        from sklearn.decomposition import SparsePCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((60, 6)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        pipe = Pipeline(
            [
                ("sp", SparsePCA(n_components=3, ridge_alpha=0.01, random_state=0)),
                ("clf", LogisticRegression()),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
