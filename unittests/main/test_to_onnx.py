"""
Unit tests for the top-level :func:`yobx.to_onnx` dispatcher.
"""

import unittest
import numpy as np

from yobx.ext_test_case import (
    ExtTestCase,
    requires_sklearn,
    requires_torch,
    skipif_ci_windows,
)
from yobx import to_onnx
from yobx.container import ExportArtifact


class TestTopLevelToOnnxSql(ExtTestCase):
    """Dispatch to :func:`yobx.sql.to_onnx` for SQL strings and callables."""

    def test_sql_string(self):
        dtypes = {"a": np.float32, "b": np.float32}
        artifact = to_onnx("SELECT a + b AS total FROM t", dtypes)
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsNotNone(artifact.proto)

    def test_callable(self):
        def transform(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        dtypes = {"a": np.float32, "b": np.float32}
        artifact = to_onnx(transform, dtypes)
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsNotNone(artifact.proto)


class TestTopLevelToOnnxSklearn(ExtTestCase):
    """Dispatch to :func:`yobx.sklearn.to_onnx` for sklearn estimators."""

    @requires_sklearn()
    def test_linear_regression(self):
        from sklearn.linear_model import LinearRegression

        X = np.random.randn(10, 3).astype(np.float32)
        y = X @ np.array([1.0, 2.0, 3.0], dtype=np.float32)
        reg = LinearRegression().fit(X, y)

        artifact = to_onnx(reg, (X,))
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsNotNone(artifact.proto)


class TestTopLevelToOnnxTorch(ExtTestCase):
    """Dispatch to :func:`yobx.torch.interpreter.to_onnx` for torch modules."""

    @requires_torch("2.5", "module 'torch.library' has no attribute 'infer_schema'")
    @skipif_ci_windows("dynamo not working on Windows")
    def test_torch_module(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)

            def forward(self, x):
                return torch.relu(self.linear(x))

        x = torch.randn(3, 4)
        artifact = to_onnx(Neuron(), (x,))
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsNotNone(artifact.proto)


class TestTopLevelToOnnxTypeError(ExtTestCase):
    """Unknown model types must raise TypeError."""

    def test_unknown_type_raises(self):
        with self.assertRaises(TypeError):
            to_onnx(42, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
