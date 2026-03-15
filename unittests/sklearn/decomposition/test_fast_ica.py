"""
Unit tests for yobx.sklearn.decomposition.FastICA converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestFastICA(ExtTestCase):
    def test_fast_ica_float32(self):
        from sklearn.decomposition import FastICA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        ica = FastICA(n_components=2, whiten="unit-variance", random_state=0)
        ica.fit(X)

        onx = to_onnx(ica, (X,))

        # Check that Sub (centering) and MatMul (projection) are present.
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)

        # Check numerical output.
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ica.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_fast_ica_float64(self):
        from sklearn.decomposition import FastICA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 4)).astype(np.float64)
        ica = FastICA(n_components=2, whiten="unit-variance", random_state=0)
        ica.fit(X)

        onx = to_onnx(ica, (X,))

        # Check that Sub (centering) and MatMul (projection) are present.
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)

        # Check numerical output.
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ica.transform(X)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-10)

    def test_fast_ica_no_whiten(self):
        from sklearn.decomposition import FastICA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        # whiten=False: no centering, components_ applied directly
        ica = FastICA(whiten=False, random_state=0)
        ica.fit(X)

        onx = to_onnx(ica, (X,))

        # Check that Sub (centering) is NOT present when whiten=False.
        op_types = [n.op_type for n in onx.graph.node]
        self.assertNotIn("Sub", op_types)
        self.assertIn("MatMul", op_types)

        # Check numerical output.
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ica.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_pipeline_fast_ica_logistic_regression(self):
        from sklearn.decomposition import FastICA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((60, 6)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        pipe = Pipeline(
            [
                ("ica", FastICA(n_components=3, whiten="unit-variance", random_state=0)),
                ("clf", LogisticRegression()),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)

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
