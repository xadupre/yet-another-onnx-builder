"""Tests the mandatory native ONNX dependency."""

import ast
import pathlib
import unittest
from unittest.mock import patch
from onnx_light import onnx


class TestOnnxShim(unittest.TestCase):
    def test_native_namespace(self):
        """Exposes only the onnx-light namespace."""
        from yobx import _onnx_shim

        self.assertIs(_onnx_shim.onnx, onnx)
        self.assertIs(_onnx_shim.helper, onnx.helper)
        self.assertTrue(onnx.ModelProto.__module__.startswith("onnx_light."))

    def test_legacy_environment_cannot_enable_reference_onnx(self):
        """Keeps native ONNX mandatory regardless of the retired switch."""
        import importlib
        from yobx import _onnx_shim

        for value in ("0", "1"):
            with self.subTest(value=value), patch.dict("os.environ", USE_OPTIM_ONNX=value):
                self.assertIs(importlib.reload(_onnx_shim).onnx, onnx)

    def test_no_reference_imports(self):
        """Rejects reference ONNX imports anywhere in the package."""
        import yobx

        violations = []
        for path in pathlib.Path(yobx.__file__).parent.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module] if isinstance(node, ast.ImportFrom) else []
                )
                if any(name and (name == "onnx" or name.startswith("onnx.")) for name in modules):
                    violations.append(f"{path}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_reference_dependent_builders_are_retired(self):
        """Rejects third-party builders instead of reinstalling reference ONNX."""
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder
        from yobx.builder.spox import SpoxGraphBuilder

        for builder in (OnnxScriptGraphBuilder, SpoxGraphBuilder):
            with (
                self.subTest(builder=builder.__name__),
                self.assertRaisesRegex(NotImplementedError, "removed reference ONNX"),
            ):
                builder(18)

    def test_python_pattern_engine_is_removed(self):
        """Prevents shipping obsolete Python matchers or their public interfaces."""
        import yobx
        import yobx.typing
        from yobx.translate import reverse_graph_builder

        package = pathlib.Path(yobx.__file__).parent
        self.assertEqual(list((package / "xoptim").rglob("*.py")), [])
        self.assertFalse(hasattr(yobx.typing, "GraphBuilderPatternOptimizationProtocol"))
        self.assertFalse(hasattr(reverse_graph_builder, "to_graph_pattern_matching"))
        retired_classes = {
            "PatternOptimization",
            "EasyPatternOptimization",
            "OnnxEasyPatternOptimization",
            "GraphBuilderPatternOptimization",
            "MatchResult",
        }
        violations = [
            f"{path}:{node.lineno}:{node.name}"
            for path in package.rglob("*.py")
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ClassDef) and node.name in retired_classes
        ]
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
