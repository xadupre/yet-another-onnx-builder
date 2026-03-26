"""
Unit tests for yobx.sklearn.ensemble.random_trees_embedding converter.
"""

import unittest
import numpy as np
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnRandomTreesEmbedding(ExtTestCase):
    def test_random_trees_embedding_float32(self):
        """Basic float32 conversion with legacy ai.onnx.ml opset."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]], dtype=np.float32
        )
        emb = RandomTreesEmbedding(n_estimators=3, max_depth=2, random_state=0)
        emb.fit(X)

        onx = to_onnx(emb, (X,), target_opset=18)

        ml_opsets = {op.domain: op.version for op in onx.proto.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        embedding = results[0]

        expected = emb.transform(X).toarray().astype(np.float32)
        self.assertEqual(embedding.dtype, np.float32)
        self.assertEqualArray(expected, embedding)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0])

    def test_random_trees_embedding_float64(self):
        """float64 input: output dtype must also be float64."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]], dtype=np.float64
        )
        emb = RandomTreesEmbedding(n_estimators=3, max_depth=2, random_state=0)
        emb.fit(X)

        onx = to_onnx(emb, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        embedding = results[0]

        expected = emb.transform(X).toarray()
        self.assertEqual(embedding.dtype, np.float64)
        self.assertEqualArray(expected, embedding)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0])

    def test_random_trees_embedding_float32_v5(self):
        """float32 with ai.onnx.ml opset 5 (TreeEnsemble operator)."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]], dtype=np.float32
        )
        emb = RandomTreesEmbedding(n_estimators=3, max_depth=2, random_state=0)
        emb.fit(X)

        onx = to_onnx(emb, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.proto.opset_import}
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        embedding = results[0]

        expected = emb.transform(X).toarray().astype(np.float32)
        self.assertEqual(embedding.dtype, np.float32)
        self.assertEqualArray(expected, embedding)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0])

    def test_random_trees_embedding_float64_v5(self):
        """float64 with ai.onnx.ml opset 5 (TreeEnsemble operator)."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]], dtype=np.float64
        )
        emb = RandomTreesEmbedding(n_estimators=3, max_depth=2, random_state=0)
        emb.fit(X)

        onx = to_onnx(emb, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.proto.opset_import}
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        embedding = results[0]

        expected = emb.transform(X).toarray()
        self.assertEqual(embedding.dtype, np.float64)
        self.assertEqualArray(expected, embedding)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0])

    def test_random_trees_embedding_single_estimator(self):
        """Edge case: n_estimators=1."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        emb = RandomTreesEmbedding(n_estimators=1, max_depth=2, random_state=0)
        emb.fit(X)

        onx = to_onnx(emb, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        expected = emb.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, results[0])

    def test_pipeline_random_trees_embedding(self):
        """RandomTreesEmbedding inside a Pipeline with a StandardScaler."""
        X = np.array(
            [[1, 2], [2, 3], [2, 4], [3, 4], [8, 9], [9, 9], [9, 10], [10, 11]], dtype=np.float32
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("emb", RandomTreesEmbedding(n_estimators=3, max_depth=2, random_state=0)),
            ]
        )
        pipe.fit(X)

        onx = to_onnx(pipe, (X,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        expected = pipe.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, results[0])

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
