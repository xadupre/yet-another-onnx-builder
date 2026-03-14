"""
Unit tests for yobx.sklearn.covariance.EllipticEnvelope converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestEllipticEnvelope(ExtTestCase):
    def _make_data(self, seed=0, n=60, n_features=4):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, n_features)).astype(np.float32)

    def _check(self, estimator, X, atol=1e-5):
        from yobx.sklearn import to_onnx

        estimator.fit(X)
        onx = to_onnx(estimator, (X,))

        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")
        self.assertEqual(output_names[0], "label")
        self.assertEqual(output_names[1], "scores")

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label_onnx, scores_onnx = results[0], results[1]

        expected_label = estimator.predict(X).astype(np.int64)
        expected_scores = estimator.decision_function(X).astype(np.float32)

        self.assertEqualArray(expected_label, label_onnx)
        self.assertEqualArray(expected_scores, scores_onnx, atol=atol)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_scores, ort_results[1], atol=atol)

    def test_elliptic_envelope_default(self):
        from sklearn.covariance import EllipticEnvelope

        X = self._make_data(seed=0)
        self._check(EllipticEnvelope(random_state=0), X)

    def test_elliptic_envelope_contamination(self):
        from sklearn.covariance import EllipticEnvelope

        X = self._make_data(seed=1)
        self._check(EllipticEnvelope(contamination=0.1, random_state=1), X)

    def test_elliptic_envelope_float64(self):
        from sklearn.covariance import EllipticEnvelope
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 3)).astype(np.float64)
        ee = EllipticEnvelope(random_state=2)
        ee.fit(X)
        onx = to_onnx(ee, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label_onnx, scores_onnx = results[0], results[1]

        expected_label = ee.predict(X).astype(np.int64)
        expected_scores = ee.decision_function(X)

        self.assertEqualArray(expected_label, label_onnx)
        self.assertEqualArray(expected_scores, scores_onnx, atol=1e-5)

    def test_elliptic_envelope_op_types(self):
        from sklearn.covariance import EllipticEnvelope
        from yobx.sklearn import to_onnx

        X = self._make_data(seed=3)
        ee = EllipticEnvelope(random_state=3)
        ee.fit(X)
        onx = to_onnx(ee, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("ReduceSum", op_types)
        self.assertIn("Where", op_types)

    def test_elliptic_envelope_pipeline(self):
        from sklearn.covariance import EllipticEnvelope
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        pipe = Pipeline([("scaler", StandardScaler()), ("ee", EllipticEnvelope(random_state=4))])
        pipe.fit(X)
        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label_onnx = results[0]

        expected_label = pipe.predict(X).astype(np.int64)
        self.assertEqualArray(expected_label, label_onnx)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
