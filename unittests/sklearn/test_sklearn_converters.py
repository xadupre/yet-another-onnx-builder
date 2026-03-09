"""
Unit tests for yobx.sklearn converters.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnBaseConverters(ExtTestCase):
    def test_standard_scaler(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        onx = to_onnx(ss, (X,))

        # Check graph structure
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)

        # Check numerical output
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ss.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_logistic_regression_binary(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        ss = StandardScaler()
        X_scaled = ss.fit_transform(X).astype(np.float32)
        lr = LogisticRegression()
        lr.fit(X_scaled, y)

        onx = to_onnx(lr, (X,))

        # Check graph structure
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gemm", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_scaled})
        label, proba = results[0], results[1]

        expected_label = lr.predict(X_scaled)
        expected_proba = lr.predict_proba(X_scaled).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_scaled})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_logistic_regression_multiclass(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        lr = LogisticRegression(max_iter=200)
        lr.fit(X, y)

        onx = to_onnx(lr, (X,))

        # Check graph structure
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gemm", op_types)
        self.assertIn("Softmax", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = lr.predict(X)
        expected_proba = lr.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_pipeline_standard_scaler_logistic_regression(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        # Check graph contains nodes from both steps
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("Gemm", op_types)

        # Check outputs
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

    @requires_sklearn("1.4")
    def test_pipeline_standard_scaler_only(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        pipe = Pipeline([("scaler", StandardScaler())])
        pipe.fit(X)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    @requires_sklearn("1.4")
    def test_pipeline_multiclass(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
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

    def test_custom_estimator_with_extra_converters(self):
        class ScaleByConstant(TransformerMixin, BaseEstimator):
            """Custom transformer that multiplies inputs by a constant."""

            def __init__(self, scale=2.0):
                self.scale = scale

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X * self.scale

        def convert_scale_by_constant(g, sts, outputs, estimator, X, name="scale"):
            import numpy as np

            scale = np.array([estimator.scale], dtype=np.float32)
            res = g.op.Mul(X, scale, name=name, outputs=outputs)
            if not sts:
                g.set_type(res, g.get_type(X))
                g.set_shape(res, g.get_shape(X))
                if g.has_device(X):
                    g.set_device(res, g.get_device(X))
            return res

        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        est = ScaleByConstant(scale=3.0)
        est.fit(X)

        onx = to_onnx(est, (X,), extra_converters={ScaleByConstant: convert_scale_by_constant})

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Mul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = est.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_extra_converters_overrides_builtin(self):
        """extra_converters entries take priority over built-in converters."""
        called = []

        def custom_scaler_converter(g, sts, outputs, estimator, X, name="scaler"):
            called.append(True)
            res = g.op.Identity(X, name=name, outputs=outputs)
            if not sts:
                g.set_type(res, g.get_type(X))
                g.set_shape(res, g.get_shape(X))
                if g.has_device(X):
                    g.set_device(res, g.get_device(X))
            return res

        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        onx = to_onnx(ss, (X,), extra_converters={StandardScaler: custom_scaler_converter})

        self.assertTrue(called, "custom converter was not called")
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Identity", op_types)
        self.assertNotIn("Sub", op_types)

    def test_estimator_without_transform_or_predict_raises(self):
        from sklearn.exceptions import NotFittedError

        class NoOpEstimator(BaseEstimator):
            def fit(self, X, y=None):
                return self

        estimator = NoOpEstimator().fit(np.zeros((4, 2), dtype=np.float32))
        X = np.zeros((4, 2), dtype=np.float32)
        with self.assertRaises(NotFittedError) as cm:
            to_onnx(estimator, (X,))
        self.assertIn("transform", str(cm.exception))
        self.assertIn("predict", str(cm.exception))

    def test_random_forest_classifier_binary(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(rf.predict(X), label)
        self.assertEqualArray(rf.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_random_forest_classifier_multiclass(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,), target_opset=18)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(rf.predict(X), label)
        self.assertEqualArray(rf.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_random_forest_regressor(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        rf = RandomForestRegressor(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,), target_opset=18)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        self.assertEqualArray(
            rf.predict(X).astype(np.float32).reshape(-1, 1), predictions, atol=1e-5
        )

    def test_random_forest_classifier_binary_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - binary classification."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(rf.predict(X), label)
        self.assertEqualArray(rf.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_random_forest_classifier_multiclass_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - multi-class classification."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(rf.predict(X), label)
        self.assertEqualArray(rf.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_random_forest_regressor_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - regression."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        rf = RandomForestRegressor(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        self.assertEqualArray(
            rf.predict(X).astype(np.float32).reshape(-1, 1), predictions, atol=1e-5
        )

    def test_pipeline_random_forest_classifier(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(pipe.predict(X), label)
        self.assertEqualArray(pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
