"""
Unit tests for yobx.sklearn converters.
"""
import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn


class TestSklearnConverters(ExtTestCase):
    """Tests for scikit-learn to ONNX converters."""

    @requires_sklearn("1.4")
    def test_standard_scaler(self):
        """Converts a StandardScaler and checks numerical output."""
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import convert_standard_scaler
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        g = convert_standard_scaler(ss)
        onx = g.to_onnx()

        # Check graph structure
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)

        # Check numerical output
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ss.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    @requires_sklearn("1.4")
    def test_logistic_regression_binary(self):
        """Converts a binary LogisticRegression and checks label and probabilities."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn import convert_logistic_regression
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        ss = StandardScaler()
        X_scaled = ss.fit_transform(X).astype(np.float32)
        lr = LogisticRegression()
        lr.fit(X_scaled, y)

        g = convert_logistic_regression(lr)
        onx = g.to_onnx()

        # Check graph structure
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gemm", op_types)
        self.assertIn("Sigmoid", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_scaled})
        label, proba = results[0], results[1]

        self.assertEqualArray(lr.predict(X_scaled), label)
        self.assertEqualArray(
            lr.predict_proba(X_scaled).astype(np.float32), proba, atol=1e-5
        )

    @requires_sklearn("1.4")
    def test_logistic_regression_multiclass(self):
        """Converts a multiclass LogisticRegression and checks label and probabilities."""
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn import convert_logistic_regression
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32
        )
        y = np.array([0, 0, 1, 1, 2, 2])
        lr = LogisticRegression(max_iter=200)
        lr.fit(X, y)

        g = convert_logistic_regression(lr)
        onx = g.to_onnx()

        # Check graph structure
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gemm", op_types)
        self.assertIn("Softmax", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(lr.predict(X), label)
        self.assertEqualArray(
            lr.predict_proba(X).astype(np.float32), proba, atol=1e-5
        )

    @requires_sklearn("1.4")
    def test_pipeline_standard_scaler_logistic_regression(self):
        """Converts a Pipeline(StandardScaler, LogisticRegression) end-to-end."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import convert_pipeline
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        g = convert_pipeline(pipe)
        onx = g.to_onnx()

        # Check graph contains nodes from both steps
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("Gemm", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(pipe.predict(X), label)
        self.assertEqualArray(
            pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5
        )

    @requires_sklearn("1.4")
    def test_pipeline_standard_scaler_only(self):
        """Converts a Pipeline with only a StandardScaler step."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import convert_pipeline
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        pipe = Pipeline([("scaler", StandardScaler())])
        pipe.fit(X)

        g = convert_pipeline(pipe)
        onx = g.to_onnx()

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    @requires_sklearn("1.4")
    def test_pipeline_multiclass(self):
        """Converts a Pipeline(StandardScaler, LogisticRegression) for multiclass."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import convert_pipeline
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32
        )
        y = np.array([0, 0, 1, 1, 2, 2])
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]
        )
        pipe.fit(X, y)

        g = convert_pipeline(pipe)
        onx = g.to_onnx()

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(pipe.predict(X), label)
        self.assertEqualArray(
            pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
