"""
Unit tests for yobx.sklearn converters.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestSklearnBaseConverters(ExtTestCase):
    def test_standard_scaler(self):
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

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

    def test_logistic_regression_binary(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn import to_onnx

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

        self.assertEqualArray(lr.predict(X_scaled), label)
        self.assertEqualArray(lr.predict_proba(X_scaled).astype(np.float32), proba, atol=1e-5)

    def test_logistic_regression_multiclass(self):
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn import to_onnx

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

        self.assertEqualArray(lr.predict(X), label)
        self.assertEqualArray(lr.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_pipeline_standard_scaler_logistic_regression(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

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

        self.assertEqualArray(pipe.predict(X), label)
        self.assertEqualArray(pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    @requires_sklearn("1.4")
    def test_pipeline_standard_scaler_only(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        pipe = Pipeline([("scaler", StandardScaler())])
        pipe.fit(X)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    @requires_sklearn("1.4")
    def test_pipeline_multiclass(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(pipe.predict(X), label)
        self.assertEqualArray(pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_decision_tree_classifier_binary(self):
        from sklearn.tree import DecisionTreeClassifier
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,))

        # Check graph structure
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleClassifier", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(dt.predict(X), label)
        self.assertEqualArray(dt.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_decision_tree_classifier_multiclass(self):
        from sklearn.tree import DecisionTreeClassifier
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,))

        # Check graph structure
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleClassifier", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(dt.predict(X), label)
        self.assertEqualArray(dt.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_decision_tree_regressor(self):
        from sklearn.tree import DecisionTreeRegressor
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,))

        # Check graph structure
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        self.assertEqualArray(
            dt.predict(X).astype(np.float32).reshape(-1, 1), predictions, atol=1e-5
        )

    def test_pipeline_decision_tree_classifier(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=0))]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        # Check graph contains nodes from both steps
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("TreeEnsembleClassifier", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(pipe.predict(X), label)
        self.assertEqualArray(pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_decision_tree_classifier_binary_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - binary classification."""
        from sklearn.tree import DecisionTreeClassifier
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        # Should use the new unified operator, not the legacy one
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        # Numerical correctness
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(dt.predict(X), label)
        self.assertEqualArray(dt.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_decision_tree_classifier_multiclass_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - multi-class classification."""
        from sklearn.tree import DecisionTreeClassifier
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(dt.predict(X), label)
        self.assertEqualArray(dt.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_decision_tree_regressor_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - regression."""
        from sklearn.tree import DecisionTreeRegressor
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        self.assertEqualArray(
            dt.predict(X).astype(np.float32).reshape(-1, 1), predictions, atol=1e-5
        )

    def test_decision_tree_legacy_opset_unchanged(self):
        """Passing an integer target_opset still emits legacy operators."""
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y_cls = np.array([0, 0, 1, 1])
        y_reg = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)

        dtc = DecisionTreeClassifier(random_state=0)
        dtc.fit(X, y_cls)
        onx_c = to_onnx(dtc, (X,), target_opset=20)
        op_types_c = [n.op_type for n in onx_c.graph.node]
        self.assertIn("TreeEnsembleClassifier", op_types_c)
        self.assertNotIn("TreeEnsemble", op_types_c)

        dtr = DecisionTreeRegressor(random_state=0)
        dtr.fit(X, y_reg)
        onx_r = to_onnx(dtr, (X,), target_opset=20)
        op_types_r = [n.op_type for n in onx_r.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types_r)
        self.assertNotIn("TreeEnsemble", op_types_r)

    def test_pipeline_decision_tree_classifier_v5(self):
        """Pipeline with TreeEnsemble (ai.onnx.ml opset 5)."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=0))]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(pipe.predict(X), label)
        self.assertEqualArray(pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_custom_estimator_with_extra_converters(self):
        from sklearn.base import BaseEstimator, TransformerMixin
        from yobx.sklearn import to_onnx

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

    def test_extra_converters_overrides_builtin(self):
        """extra_converters entries take priority over built-in converters."""
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
