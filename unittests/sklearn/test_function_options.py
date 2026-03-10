"""
Unit tests for the SklearnFunctionOptions feature of yobx.sklearn.to_onnx.

When ``function_options`` is provided every non-container estimator is
exported as a separate ONNX local function.  Pipeline and ColumnTransformer
are treated as orchestrators — their individual steps/sub-transformers each
become a local function.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from yobx.sklearn import SklearnFunctionOptions, to_onnx


@requires_sklearn("1.4")
class TestSklearnFunctionOptions(ExtTestCase):
    """Tests for wrapping sklearn estimators as ONNX local functions."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.standard_normal((10, 4)).astype(np.float32)
        self.y = (self.X[:, 0] > 0).astype(int)
        self.fopts = SklearnFunctionOptions(domain="test_sklearn")

    # ------------------------------------------------------------------
    # SklearnFunctionOptions construction
    # ------------------------------------------------------------------

    def test_sklearn_function_options_repr(self):
        opts = SklearnFunctionOptions(domain="my_domain")
        self.assertIn("my_domain", repr(opts))

    def test_sklearn_function_options_empty_domain_raises(self):
        with self.assertRaises(ValueError):
            SklearnFunctionOptions(domain="")

    # ------------------------------------------------------------------
    # Standalone estimator → wrapped as a single local function
    # ------------------------------------------------------------------

    def test_standard_scaler_as_function(self):
        ss = StandardScaler().fit(self.X)
        onx = to_onnx(ss, (self.X,), function_options=self.fopts)

        # One local function with the estimator's class name.
        func_names = [(f.name, f.domain) for f in onx.functions]
        self.assertEqual(len(func_names), 1)
        self.assertIn(("StandardScaler", "test_sklearn"), func_names)

        # The main graph has a single function-call node.
        graph_ops = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("StandardScaler", "test_sklearn"), graph_ops)
        self.assertNotIn("Sub", [n.op_type for n in onx.graph.node])
        self.assertNotIn("Div", [n.op_type for n in onx.graph.node])

        # Numerical correctness.
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": self.X})[0]
        expected = ss.transform(self.X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_logistic_regression_as_function(self):
        lr = LogisticRegression(max_iter=200).fit(self.X, self.y)
        onx = to_onnx(lr, (self.X,), function_options=self.fopts)

        func_names = [f.name for f in onx.functions]
        self.assertIn("LogisticRegression", func_names)

        ref = ExtendedReferenceEvaluator(onx)
        label, proba = ref.run(None, {"X": self.X})
        self.assertEqualArray(lr.predict(self.X), label)
        self.assertEqualArray(
            lr.predict_proba(self.X).astype(np.float32), proba, atol=1e-5
        )

    # ------------------------------------------------------------------
    # Without function_options → backward-compatible flat graph
    # ------------------------------------------------------------------

    def test_no_function_options_produces_flat_graph(self):
        ss = StandardScaler().fit(self.X)
        onx = to_onnx(ss, (self.X,))
        # No local functions when option is not provided.
        self.assertEqual(len(onx.functions), 0)
        # Raw ops are present in the graph.
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)

    # ------------------------------------------------------------------
    # Pipeline → each step is a separate local function
    # ------------------------------------------------------------------

    def test_pipeline_steps_as_functions(self):
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]
        ).fit(self.X, self.y)
        onx = to_onnx(pipe, (self.X,), function_options=self.fopts)

        func_names = [f.name for f in onx.functions]
        # Both steps must appear as local functions.
        self.assertIn("StandardScaler", func_names)
        self.assertIn("LogisticRegression", func_names)
        # The pipeline container itself is NOT a function.
        self.assertNotIn("Pipeline", func_names)

        # No raw ops from the individual converters in the main graph.
        graph_ops = [n.op_type for n in onx.graph.node]
        self.assertNotIn("Sub", graph_ops)
        self.assertNotIn("Gemm", graph_ops)

        # Numerical correctness.
        ref = ExtendedReferenceEvaluator(onx)
        label, proba = ref.run(None, {"X": self.X})
        self.assertEqualArray(pipe.predict(self.X), label)
        self.assertEqualArray(
            pipe.predict_proba(self.X).astype(np.float32), proba, atol=1e-5
        )

    def test_pipeline_not_wrapped_without_function_options(self):
        """Pipeline without function_options keeps the flat graph."""
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]
        ).fit(self.X, self.y)
        onx = to_onnx(pipe, (self.X,))
        self.assertEqual(len(onx.functions), 0)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Gemm", op_types)

    # ------------------------------------------------------------------
    # ColumnTransformer → each sub-transformer is a local function
    # ------------------------------------------------------------------

    def test_column_transformer_transformers_as_functions(self):
        X = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32
        )
        ct = ColumnTransformer(
            [("std", StandardScaler(), [0, 1]), ("mms", MinMaxScaler(), [2, 3])]
        ).fit(X)
        onx = to_onnx(ct, (X,), function_options=self.fopts)

        func_names = [f.name for f in onx.functions]
        self.assertIn("StandardScaler", func_names)
        self.assertIn("MinMaxScaler", func_names)
        self.assertNotIn("ColumnTransformer", func_names)

        # Concat is still in the main graph (CT orchestration).
        graph_ops = [n.op_type for n in onx.graph.node]
        self.assertIn("Concat", graph_ops)
        self.assertNotIn("Sub", graph_ops)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ct.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    # ------------------------------------------------------------------
    # Custom domain is preserved
    # ------------------------------------------------------------------

    def test_custom_domain_is_used(self):
        ss = StandardScaler().fit(self.X)
        custom_opts = SklearnFunctionOptions(domain="acme.corp.v1")
        onx = to_onnx(ss, (self.X,), function_options=custom_opts)

        self.assertEqual(len(onx.functions), 1)
        self.assertEqual(onx.functions[0].domain, "acme.corp.v1")

    # ------------------------------------------------------------------
    # SklearnFunctionOptions is exported from the sklearn package
    # ------------------------------------------------------------------

    def test_import_from_sklearn_package(self):
        from yobx.sklearn import SklearnFunctionOptions as SFO

        self.assertIs(SFO, SklearnFunctionOptions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
