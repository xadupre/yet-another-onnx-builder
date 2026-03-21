"""
Unit tests for the RegressorChain ONNX converter.
"""

import unittest
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import RegressorChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnRegressorChain(ExtTestCase):
    """Tests for the RegressorChain converter."""

    _X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    _y = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0]])

    def test_float32_default_order(self):
        """RegressorChain with Ridge, default order, float32 input."""
        rc = RegressorChain(Ridge())
        rc.fit(self._X, self._y)

        onx = to_onnx(rc, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        predictions = results[0]

        expected = rc.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_float64_default_order(self):
        """RegressorChain with Ridge, default order, float64 input."""
        X64 = self._X.astype(np.float64)
        y64 = self._y.astype(np.float64)
        rc = RegressorChain(Ridge())
        rc.fit(X64, y64)

        onx = to_onnx(rc, (X64,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X64})
        predictions = results[0]

        expected = rc.predict(X64)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X64})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_float32_custom_order(self):
        """RegressorChain with custom order, float32 input."""
        rc = RegressorChain(Ridge(), order=[2, 0, 1])
        rc.fit(self._X, self._y)

        onx = to_onnx(rc, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        predictions = results[0]

        expected = rc.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_float64_custom_order(self):
        """RegressorChain with custom order, float64 input."""
        X64 = self._X.astype(np.float64)
        y64 = self._y.astype(np.float64)
        rc = RegressorChain(Ridge(), order=[2, 0, 1])
        rc.fit(X64, y64)

        onx = to_onnx(rc, (X64,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X64})
        predictions = results[0]

        expected = rc.predict(X64)
        self.assertEqualArray(expected, predictions, atol=1e-5)

    def test_two_targets(self):
        """RegressorChain with 2 targets."""
        y2 = self._y[:, :2]
        rc = RegressorChain(Ridge())
        rc.fit(self._X, y2)

        onx = to_onnx(rc, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        predictions = results[0]

        expected = rc.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)
        self.assertEqual(predictions.shape, (4, 2))

    def test_decision_tree_regressor(self):
        """RegressorChain wrapping DecisionTreeRegressor."""
        rc = RegressorChain(DecisionTreeRegressor(max_depth=2, random_state=0))
        rc.fit(self._X, self._y)

        onx = to_onnx(rc, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        predictions = results[0]

        expected = rc.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

    def test_pipeline_with_regressor_chain(self):
        """RegressorChain at the end of a Pipeline."""
        pipe = Pipeline([("scaler", StandardScaler()), ("rc", RegressorChain(Ridge()))])
        pipe.fit(self._X, self._y)

        onx = to_onnx(pipe, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        predictions = results[0]

        expected = pipe.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

    def test_graph_structure_default_order(self):
        """Check ONNX graph structure for default-order RegressorChain."""
        rc = RegressorChain(Ridge())
        rc.fit(self._X, self._y)

        onx = to_onnx(rc, (self._X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        # Three sub-estimators → three Gemm nodes (Ridge → Gemm)
        self.assertEqual(op_types.count("Gemm"), 3)
        # Three Reshape ops (one per chain step)
        self.assertEqual(op_types.count("Reshape"), 3)
        # No Gather needed for identity order
        self.assertNotIn("Gather", op_types)

    def test_graph_structure_custom_order(self):
        """Check ONNX graph structure for custom-order RegressorChain (Gather present)."""
        rc = RegressorChain(Ridge(), order=[2, 0, 1])
        rc.fit(self._X, self._y)

        onx = to_onnx(rc, (self._X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        # Three sub-estimators → three Gemm nodes
        self.assertEqual(op_types.count("Gemm"), 3)
        # A Gather node reorders the columns
        self.assertIn("Gather", op_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
