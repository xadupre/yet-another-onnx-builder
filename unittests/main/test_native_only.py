"""Exercises the default export lifecycle while reference ONNX imports are forbidden."""

import os
import subprocess
import sys
import textwrap
import unittest


class TestNativeOnly(unittest.TestCase):
    def test_export_save_load_and_report_without_reference_onnx(self):
        """Builds, optimizes, reports, saves and reloads native protobufs."""
        code = textwrap.dedent("""
            import importlib.abc
            import importlib.util
            import pathlib
            import sys
            import tempfile

            class NoReferenceOnnx(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "onnx" or fullname.startswith("onnx."):
                        raise AssertionError(f"Reference ONNX import: {fullname}")
                    return None

            sys.meta_path.insert(0, NoReferenceOnnx())
            from onnx_light import onnx
            from onnx_light.onnx_core.graph_builder import GraphBuilder as NativeBuilder
            from yobx.xbuilder import GraphBuilder, OptimizationOptions
            from yobx.xbuilder.graph_builder_opset import Opset
            from yobx.container import ExportArtifact
            import yobx.xshape

            assert not hasattr(yobx.xshape, "Basic" + "ShapeBuilder")
            builder = GraphBuilder(
                18, ir_version=8,
                optimization_options=OptimizationOptions(patterns=["TransposeTranspose"])
            )
            assert isinstance(builder.inner_builder, NativeBuilder)
            assert type(builder.op) is Opset
            assert importlib.util.find_spec("yobx.xbuilder.order_optim") is None
            builder.make_tensor_input("X", onnx.TensorProto.FLOAT, ("batch", 3))
            transposed = builder.op.Transpose("X", perm=[1, 0])
            output = builder.op.Transpose(transposed, perm=[1, 0])
            builder.make_tensor_output(output)
            artifact = builder.to_onnx(return_optimize_report=True)
            assert isinstance(artifact.proto, onnx.ModelProto)
            assert artifact.report.extra["rewrites"] == 1
            assert len(artifact.proto.graph.node) == 1
            with tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "native.onnx"
                artifact.save(str(path))
                assert path.with_suffix(".xlsx").is_file()
                loaded = ExportArtifact.load(str(path))
                assert isinstance(loaded.proto, onnx.ModelProto)
                assert loaded.proto.SerializeToString() == artifact.proto.SerializeToString()
            assert not any(n == "onnx" or n.startswith("onnx.") for n in sys.modules)
            """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={
                **os.environ,
                "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
