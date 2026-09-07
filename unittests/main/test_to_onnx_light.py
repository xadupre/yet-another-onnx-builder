"""Exercises the native backend through the public conversion entry point."""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from yobx import to_onnx
from yobx.ext_test_case import requires_sklearn


class TestGraphBackendSelection(unittest.TestCase):
    def test_unknown_backend(self):
        """Rejects unknown backends before importing converter dependencies."""
        with self.assertRaisesRegex(ValueError, "Unknown graph_backend"):
            to_onnx(object(), graph_backend="missing")

    def test_conflicting_builder(self):
        """Rejects an ambiguous builder selection."""
        with self.assertRaisesRegex(ValueError, "cannot be specified together"):
            to_onnx(object(), graph_backend="onnx-light", builder_cls=object)

    def test_numpy_builder_factory(self):
        """Uses a supplied builder factory for NumPy tracing."""
        from yobx.sql import to_onnx as sql_to_onnx
        from yobx.xbuilder import GraphBuilder

        x = np.array([1, 2], dtype=np.float32)
        factory = Mock(wraps=GraphBuilder)
        artifact = sql_to_onnx(lambda a: a + 1, x, builder_cls=factory)
        factory.assert_called_once()
        self.assertIsNotNone(artifact.proto)

    def test_torch_native_dispatch(self):
        """Dispatches both default and explicit native Torch requests."""
        import sys
        import types

        class Module:
            pass

        torch_module = types.ModuleType("torch")
        torch_module.nn = types.SimpleNamespace(Module=Module)
        torch_module.fx = types.SimpleNamespace(GraphModule=Module)
        converter_module = types.ModuleType("yobx.torch")
        converter_module.to_onnx = Mock()
        with (
            patch("yobx.ext_test_case.has_torch", return_value=True),
            patch.dict(sys.modules, {"torch": torch_module, "yobx.torch": converter_module}),
        ):
            for backend in (None, "onnx-light"):
                with self.subTest(backend=backend):
                    model = Module()
                    converter_module.to_onnx.reset_mock()
                    result = to_onnx(model, graph_backend=backend)
                    self.assertIs(result, converter_module.to_onnx.return_value)
                    converter_module.to_onnx.assert_called_once()
                    self.assertEqual(converter_module.to_onnx.call_args.args, (model, None))
                    self.assertNotIn("graph_backend", converter_module.to_onnx.call_args.kwargs)


class TestNativeBackendConversion(unittest.TestCase):
    def run_model(self, artifact, feeds):
        """Runs the serialized model with ONNX Runtime."""
        from onnxruntime import InferenceSession

        session = InferenceSession(
            artifact.proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return session.run(None, feeds)

    def test_sql(self):
        """Converts SQL using the native builder instead of the Python builder."""
        from onnx_light.onnx_core.graph_builder import GraphBuilder

        artifact = to_onnx(
            "SELECT a + b AS total FROM t",
            {"a": np.float32, "b": np.float32},
            return_optimize_report=True,
        )
        self.assertIsInstance(artifact.builder.inner_builder, GraphBuilder)
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        actual = self.run_model(artifact, {"a": a, "b": b})
        np.testing.assert_allclose(actual[0], a + b)
        self.assertIsNotNone(artifact.report)
        self.assertEqual(artifact.report.extra["backend"], "onnx-light")
        self.assertGreaterEqual(artifact.report.extra["rewrites"], 0)

    def test_numpy_callable(self):
        """Propagates the native builder through the NumPy tracing dispatcher."""

        def transform(x):
            return np.sqrt(np.abs(x) + np.float32(1))

        x = np.array([-1, 2, 3], dtype=np.float32)
        artifact = to_onnx(transform, x, graph_backend="onnx-light")
        actual = self.run_model(artifact, {"X": x})
        np.testing.assert_allclose(actual[0], transform(x), rtol=1e-6)

    def test_dataframe_report(self):
        """Propagates native reports through parsed DataFrame queries."""

        def transform(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        artifact = to_onnx(
            transform,
            {"a": np.float32, "b": np.float32},
            graph_backend="onnx-light",
            return_optimize_report=True,
        )
        self.assertEqual(artifact.report.extra["backend"], "onnx-light")
        a = np.array([1, 2], dtype=np.float32)
        np.testing.assert_array_equal(self.run_model(artifact, {"a": a, "b": a})[0], 2 * a)

    def test_litert_report(self):
        """Preserves native reports when finalizing LiteRT conversion."""
        from yobx.builder.onnxlight import OnnxLightGraphBuilder
        from yobx.litert import to_onnx as litert_to_onnx
        from yobx.litert.litert_helper import _make_sample_tflite_model

        artifact = litert_to_onnx(
            _make_sample_tflite_model(),
            input_names=["X"],
            target_opset={"": 18, "com.microsoft": 1},
            builder_cls=OnnxLightGraphBuilder,
            return_optimize_report=True,
        )
        self.assertEqual(artifact.report.extra["backend"], "onnx-light")
        x = np.array([[-1, 0, 1, 2]], dtype=np.float32)
        np.testing.assert_array_equal(self.run_model(artifact, {"X": x})[0], np.maximum(x, 0))

    def test_save_artifact(self):
        """Saves native-built models through the existing artifact interface."""
        import os
        import tempfile
        from yobx._onnx_shim import onnx

        with tempfile.TemporaryDirectory() as folder:
            filename = os.path.join(folder, "model.onnx")
            artifact = to_onnx(
                "SELECT a + b AS total FROM t",
                {"a": np.float32, "b": np.float32},
                graph_backend="onnx-light",
                filename=filename,
            )
            self.assertEqual(
                onnx.load(filename).SerializeToString(), artifact.SerializeToString()
            )

    @requires_sklearn("1.4")
    def test_sklearn(self):
        """Converts a fitted regression and preserves its dynamic batch dimension."""
        from sklearn.linear_model import LinearRegression

        x = np.random.default_rng(0).standard_normal((20, 4)).astype(np.float32)
        model = LinearRegression().fit(x, x[:, 0] + 2 * x[:, 1])
        artifact = to_onnx(
            model,
            (x,),
            input_names=["X"],
            dynamic_shapes=({0: "batch"},),
            target_opset={"": 18, "com.microsoft": 1},
            graph_backend="onnx-light",
            return_optimize_report=True,
        )
        self.assertEqual(
            artifact.proto.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch"
        )
        actual = self.run_model(artifact, {"X": x[:3]})
        np.testing.assert_allclose(actual[0].reshape(-1), model.predict(x[:3]), rtol=1e-5)
        self.assertIsNotNone(artifact.report)
        self.assertEqual(artifact.report.extra["backend"], "onnx-light")


if __name__ == "__main__":
    unittest.main(verbosity=2)
