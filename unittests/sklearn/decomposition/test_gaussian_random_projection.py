"""
Unit tests for yobx.sklearn.decomposition.GaussianRandomProjection converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestGaussianRandomProjection(ExtTestCase):
    def test_gaussian_random_projection_basic(self):
        from sklearn.random_projection import GaussianRandomProjection
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 10)).astype(np.float32)
        proj = GaussianRandomProjection(n_components=4, random_state=0)
        proj.fit(X)

        onx = to_onnx(proj, (X,))

        # Check that a MatMul (projection) node is present.
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)

        # Check numerical output.
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = proj.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_gaussian_random_projection_float64(self):
        from sklearn.random_projection import GaussianRandomProjection
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((15, 8)).astype(np.float64)
        proj = GaussianRandomProjection(n_components=3, random_state=1)
        proj.fit(X)

        onx = to_onnx(proj, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = proj.transform(X)
        self.assertEqualArray(expected, result, atol=1e-10)

    def test_pipeline_gaussian_random_projection(self):
        from sklearn.random_projection import GaussianRandomProjection
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 8)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        pipe = Pipeline(
            [
                ("proj", GaussianRandomProjection(n_components=4, random_state=0)),
                ("clf", LogisticRegression()),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
