"""
Unit tests for the top-level :func:`yobx.to_onnx` dispatcher.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_torch


class TestToOnnxDispatcher(ExtTestCase):
    """Verify that :func:`yobx.to_onnx` dispatches to the correct backend."""

    def test_unsupported_type_raises(self):
        """Passing an unsupported object should raise TypeError."""
        from yobx import to_onnx

        with self.assertRaises(TypeError):
            to_onnx(42, None)

    @requires_sklearn()
    def test_dispatch_sklearn(self):
        """A fitted sklearn estimator is dispatched to yobx.sklearn.to_onnx."""
        from sklearn.linear_model import LinearRegression
        from yobx import to_onnx
        from yobx.container import ExportArtifact

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = X[:, 0] + X[:, 1]
        model = LinearRegression().fit(X, y)

        artifact = to_onnx(model, (X,))
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsNotNone(artifact.proto)

    def test_dispatch_sql_string(self):
        """A plain SQL string is dispatched to yobx.sql.to_onnx."""
        from yobx import to_onnx
        from yobx.container import ExportArtifact

        artifact = to_onnx("SELECT a + b AS total FROM t", {"a": np.float32, "b": np.float32})
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsNotNone(artifact.proto)

    def test_dispatch_callable(self):
        """A callable (numpy function) is dispatched to yobx.sql.to_onnx."""
        from yobx import to_onnx
        from yobx.container import ExportArtifact

        def my_func(x):
            return x + 1.0

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        artifact = to_onnx(my_func, x)
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsNotNone(artifact.proto)

    @requires_torch()
    def test_dispatch_torch(self):
        """A torch.nn.Module is dispatched to yobx.torch.to_onnx."""
        import torch
        from yobx import to_onnx
        from yobx.container import ExportArtifact

        class Neuron(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)

            def forward(self, x):
                return torch.relu(self.linear(x))

        model = Neuron()
        x = torch.randn(3, 4)
        artifact = to_onnx(model, (x,))
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsNotNone(artifact.proto)


class TestToOnnxReturnOptimizeReport(ExtTestCase):
    """Verify that :func:`yobx.to_onnx` propagates *return_optimize_report* to backends."""

    def test_sql_default_report_is_none(self):
        """Report is None when return_optimize_report is not set (default False)."""
        from yobx import to_onnx

        artifact = to_onnx(
            "SELECT a + b AS total FROM t", {"a": np.float32, "b": np.float32}
        )
        self.assertIsNone(artifact.report)

    def test_sql_report_populated_when_true(self):
        """Report is an ExportReport with stats when return_optimize_report=True."""
        from yobx import to_onnx
        from yobx.container import ExportReport

        artifact = to_onnx(
            "SELECT a + b AS total FROM t",
            {"a": np.float32, "b": np.float32},
            return_optimize_report=True,
        )
        self.assertIsNotNone(artifact.report)
        self.assertIsInstance(artifact.report, ExportReport)
        self.assertIsInstance(artifact.report.stats, list)
        self.assertGreater(len(artifact.report.stats), 0)

    def test_callable_default_report_is_none(self):
        """Report is None by default for callable (numpy function) dispatch."""
        from yobx import to_onnx

        def my_func(x):
            return x + 1.0

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        artifact = to_onnx(my_func, x)
        self.assertIsNone(artifact.report)

    def test_callable_report_populated_when_true(self):
        """Report is populated for callable dispatch with return_optimize_report=True."""
        from yobx import to_onnx
        from yobx.container import ExportReport

        def my_func(x):
            return x + 1.0

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        artifact = to_onnx(my_func, x, return_optimize_report=True)
        self.assertIsNotNone(artifact.report)
        self.assertIsInstance(artifact.report, ExportReport)
        self.assertGreater(len(artifact.report.stats), 0)

    @requires_sklearn()
    def test_sklearn_default_report_is_none(self):
        """Report is None by default for sklearn dispatch."""
        from sklearn.linear_model import LinearRegression
        from yobx import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = X[:, 0] + X[:, 1]
        model = LinearRegression().fit(X, y)
        artifact = to_onnx(model, (X,))
        self.assertIsNone(artifact.report)

    @requires_sklearn()
    def test_sklearn_report_populated_when_true(self):
        """Report is populated for sklearn dispatch with return_optimize_report=True."""
        from sklearn.linear_model import LinearRegression
        from yobx import to_onnx
        from yobx.container import ExportReport

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = X[:, 0] + X[:, 1]
        model = LinearRegression().fit(X, y)
        artifact = to_onnx(model, (X,), return_optimize_report=True)
        self.assertIsNotNone(artifact.report)
        self.assertIsInstance(artifact.report, ExportReport)
        self.assertGreater(len(artifact.report.stats), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
