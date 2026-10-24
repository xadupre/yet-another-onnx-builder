"""Tests Python-file export discovery."""

import tempfile
import textwrap
import unittest
from pathlib import Path
from yobx import guess_export, to_onnx
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_torch


class TestGuessExport(ExtTestCase):
    def discover(self, source, **kwargs):
        """Discovers export arguments from a temporary Python file."""
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "model.py"
            filename.write_text(textwrap.dedent(source), encoding="utf-8")
            return guess_export(filename, **kwargs)

    def test_callable(self):
        result = self.discover("""
            from math import sin
            class Unused:
                def __init__(self):
                    raise AssertionError("Must not instantiate classes")
            def _helper(x):
                return x
            def predict(x):
                return x + 1
            alias = predict
            x = 2
            input_names = ["data"]
            dynamic_shapes = ({0: "batch"},)
            if __name__ == "__main__":
                raise AssertionError("Must not execute main")
            """)
        self.assertEqual(result["model"](*result["args"]), 3)
        self.assertEqual(result["args"], (2,))
        self.assertEqual(result["input_names"], ["data"])
        self.assertEqual(result["dynamic_shapes"], ({0: "batch"},))

    def test_model_and_input_priority(self):
        result = self.discover("""
            def helper(x):
                return x
            model = lambda x: x + 2
            args = (3,)
            example_inputs = (4,)
            inputs = (5,)
            X = 6
            x = 7
            """)
        self.assertEqual(result["model"](*result["args"]), 5)

    def test_explicit_names(self):
        result = self.discover(
            """
            from math import sin
            def other(x):
                return x
            samples = [0]
            """,
            model_name="sin",
            args_name="samples",
        )
        self.assertEqual(result["model"](*result["args"]), 0)
        self.assertEqual(result["args"], [0])

    def test_sql_mapping(self):
        result = self.discover("model = 'SELECT a FROM t'\ninputs = {'a': 'float32'}")
        self.assertEqual(result["args"], {"a": "float32"})

    def test_input_aliases(self):
        for name in ("args", "example_inputs", "inputs", "X", "x"):
            with self.subTest(name=name):
                result = self.discover(f"model = lambda x: x\n{name} = (1, 2)")
                self.assertEqual(result["args"], (1, 2))
        result = self.discover("model = lambda: 1\nargs = ()")
        self.assertEqual(result["args"], ())

    def test_ambiguity_and_missing_objects(self):
        cases = [
            ("x = 1", {}, "identify one model"),
            ("from math import sin\nx = 1", {}, "identify one model"),
            ("class Model:\n    pass\nx = 1", {}, "identify one model"),
            ("def a(x): return x\ndef b(x): return x\nx = 1", {}, "model_name"),
            ("model = lambda x: x", {}, r"candidates=\[\].*Specify args_name"),
            (
                "model = lambda x: x\nX = 1\nx = 2",
                {},
                r"candidates=\['X', 'x'\].*Specify args_name",
            ),
            ("model = lambda x: x\nargs = None", {}, "is None"),
            ("x = 1", {"model_name": "missing"}, "Model variable"),
            ("model = lambda x: x", {"args_name": "missing"}, "Input variable"),
        ]
        for source, kwargs, message in cases:
            with (
                self.subTest(source=source, kwargs=kwargs),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.discover(source, **kwargs)
        for source in ("model = 42\nx = 1", "model = int\nx = 1"):
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(TypeError, "unsupported model type"),
            ):
                self.discover(source)

    def test_execution_errors(self):
        with self.assertRaisesRegex(RuntimeError, "script failure"):
            self.discover("raise RuntimeError('script failure')")
        with self.assertRaises(SyntaxError):
            self.discover("this is not valid python!")
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(FileNotFoundError):
            guess_export(Path(folder) / "missing.py")

    def test_export_numpy(self):
        result = self.discover("""
            import numpy as np
            def predict(x):
                return x + np.float32(1)
            x = np.array([[1, 2]], dtype=np.float32)
            input_names = ["data"]
            """)
        artifact = to_onnx(**result, target_opset=18)
        self.assertEqual([v.name for v in artifact.proto.graph.input], ["data"])
        self.assertTrue(artifact.proto.graph.node)

    @requires_sklearn()
    def test_sklearn(self):
        result = self.discover("""
            import numpy as np
            from sklearn.linear_model import LinearRegression
            X = np.arange(12, dtype=np.float32).reshape(4, 3)
            reg = LinearRegression().fit(X, X.sum(axis=1))
            """)
        self.assertEqual(type(result["model"]).__name__, "LinearRegression")
        self.assertEqual(result["args"][0].shape, (4, 3))
        self.assertTrue(to_onnx(**result, target_opset=18).proto.graph.node)

    @requires_torch()
    def test_torch(self):
        result = self.discover("""
            import torch
            class Neuron(torch.nn.Module):
                def forward(self, x):
                    return x + 1
            net = Neuron()
            example_inputs = (torch.ones(2, 3),)
            """)
        self.assertEqual(type(result["model"]).__name__, "Neuron")
        self.assertEqual(tuple(result["args"][0].shape), (2, 3))
        self.assertTrue(to_onnx(**result, target_opset=18).proto.graph.node)


if __name__ == "__main__":
    unittest.main()
