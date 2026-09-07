"""Exercises the native Torch conversion path without the legacy ONNX engines."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy
from onnx_light import onnx
from onnx_light.onnx import helper


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires the published torch wheel.")
class TestNativeTorchExport(unittest.TestCase):
    def session(self, model):
        from onnxruntime import InferenceSession, SessionOptions

        options = SessionOptions()
        options.log_severity_level = 3
        return InferenceSession(
            model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def test_public_builder_imports_are_native(self):
        from yobx.builder.onnxlight import OnnxLightGraphBuilder, OnnxLightOptimizationOptions
        from yobx.xbuilder import GraphBuilder
        from yobx.xbuilder.graph_builder import GraphBuilder as DirectGraphBuilder
        from yobx.xbuilder.optimization_options import OptimizationOptions

        self.assertIs(GraphBuilder, OnnxLightGraphBuilder)
        self.assertIs(DirectGraphBuilder, OnnxLightGraphBuilder)
        self.assertIs(OptimizationOptions, OnnxLightOptimizationOptions)
        with self.assertRaises(TypeError):
            OptimizationOptions(constant_folding=False)

    def test_linear_relu_dynamic_batch(self):
        import torch
        from onnx_light.onnx_core.graph_builder import GraphBuilder
        from onnx_light.onnx_core.shape_inference import ShapesContext
        from yobx.torch import to_onnx

        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(inplace=True)).eval()
        artifact = to_onnx(
            model,
            (torch.randn(2, 4),),
            input_names=["X"],
            output_names=["Y"],
            dynamic_shapes=({0: "batch"},),
            return_builder=True,
            return_ep=True,
            return_optimize_report=True,
        )
        self.assertIsInstance(artifact.proto, onnx.ModelProto)
        self.assertIsInstance(artifact.builder.inner_builder, GraphBuilder)
        self.assertIsInstance(artifact.builder.shapes_context, ShapesContext)
        self.assertIsInstance(artifact.ep, torch.export.ExportedProgram)
        self.assertEqual(artifact.report.extra["backend"], "onnx-light")
        self.assertEqual(artifact.report.extra["torch_backend"], "native")
        self.assertEqual(artifact.builder.get_shape("Y"), ("batch", 3))
        self.assertEqual(
            artifact.proto.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch"
        )
        session = self.session(artifact.proto)
        for batch in (1, 2, 7):
            x = torch.randn(batch, 4)
            actual = session.run(None, {"X": x.numpy()})[0]
            numpy.testing.assert_allclose(actual, model(x).detach().numpy(), rtol=1e-5, atol=1e-6)

    def test_linear_vector_and_batched_double(self):
        import torch
        from yobx.torch import to_onnx

        model = torch.nn.Linear(4, 3).double().eval()
        for shape in ((4,), (2, 5, 4)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.float64)
                artifact = to_onnx(model, (x,), input_names=["X"])
                actual = self.session(artifact.proto).run(None, {"X": x.numpy()})[0]
                numpy.testing.assert_allclose(
                    actual, model(x).detach().numpy(), rtol=1e-12, atol=1e-12
                )

    def test_exported_input_signature_errors(self):
        """Reports unsupported symbolic arguments and missing parameter targets."""
        import torch
        from torch.export.graph_signature import InputKind, SymIntArgument
        from yobx.torch import to_onnx
        from yobx.torch.export_options import ExportOptions

        class Model(torch.nn.Module):
            def forward(self, x, scale):
                return x * scale

        x = torch.randn(2, 3)
        options = ExportOptions(remove_inplace=False)
        program = torch.export.export(Model(), (x, 2))
        artifact = to_onnx(program, validate_onnx=True, export_options=options)
        self.assertIsNone(artifact.ep)
        spec = program.graph_signature.input_specs[-1]
        spec.arg = SymIntArgument(name=spec.arg.name)
        with self.assertRaisesRegex(NotImplementedError, "does not support user input"):
            to_onnx(program, export_options=options)

        program = torch.export.export(torch.nn.Linear(3, 2), (x,))
        spec = next(
            spec
            for spec in program.graph_signature.input_specs
            if spec.kind == InputKind.PARAMETER
        )
        spec.target = None
        with self.assertRaisesRegex(ValueError, "has no target"):
            to_onnx(program, export_options=options)

    def test_native_transpose_pattern(self):
        import torch
        from yobx.builder.onnxlight import OnnxLightOptimizationOptions
        from yobx.torch import to_onnx

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.transpose(0, 1).transpose(0, 1)

        options = OnnxLightOptimizationOptions(["TransposeTranspose"])
        x = torch.randn(2, 3)
        plain = to_onnx(Model(), (x,), input_names=["X"], options=options, optimize=False)
        optimized = to_onnx(Model(), (x,), input_names=["X"], options=options)
        self.assertEqual(
            [str(n.op_type) for n in plain.proto.graph.node], ["Transpose", "Transpose"]
        )
        self.assertEqual([str(n.op_type) for n in optimized.proto.graph.node], ["Identity"])
        self.assertEqual(optimized.report.extra["rewrites"], 1)
        self.assertEqual(optimized.report.stats[0]["pattern"], "TransposeTranspose")
        numpy.testing.assert_array_equal(
            self.session(optimized.proto).run(None, {"X": x.numpy()})[0], x.numpy()
        )

    def test_dynamic_reshape_reduction(self):
        import torch
        from yobx.torch import to_onnx

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.reshape(x.shape[0], -1).sum(dim=1)

        artifact = to_onnx(
            Model(),
            (torch.randn(2, 3, 4),),
            input_names=["X"],
            dynamic_shapes=({0: torch.export.Dim("batch")},),
        )
        self.assertEqual(
            artifact.proto.graph.output[0].type.tensor_type.shape.dim[0].dim_param, "batch"
        )
        session = self.session(artifact.proto)
        for batch in (1, 5):
            x = torch.randn(batch, 3, 4)
            numpy.testing.assert_allclose(
                session.run(None, {"X": x.numpy()})[0], Model()(x).numpy(), rtol=1e-5, atol=1e-6
            )

    def test_conditional_branches_and_symbolic_predicate(self):
        import torch
        from yobx.torch import to_onnx

        class Conditional(torch.nn.Module):
            def forward(self, predicate, x):
                return torch.cond(
                    predicate, lambda value: value + 1, lambda value: value - 1, (x,)
                )

        artifact = to_onnx(
            Conditional(),
            (torch.tensor(True), torch.randn(2, 3)),
            input_names=["predicate", "X"],
            dynamic_shapes=(None, {0: "batch"}),
        )
        conditional = next(node for node in artifact.proto.graph.node if node.op_type == "If")
        self.assertTrue(all(len(attribute.g.input) == 0 for attribute in conditional.attribute))
        session = self.session(artifact.proto)
        for predicate in (True, False):
            x = numpy.arange(15, dtype=numpy.float32).reshape(5, 3)
            actual = session.run(None, {"predicate": numpy.array(predicate), "X": x})[0]
            numpy.testing.assert_array_equal(actual, x + (1 if predicate else -1))

        class SymbolicConditional(torch.nn.Module):
            def forward(self, x):
                return torch.cond(
                    x.shape[0] > 3, lambda value: value * 2, lambda value: value - 2, (x,)
                )

        artifact = to_onnx(
            SymbolicConditional(),
            (torch.randn(2, 3),),
            input_names=["X"],
            dynamic_shapes=({0: "batch"},),
        )
        session = self.session(artifact.proto)
        for batch in (2, 5):
            x = numpy.ones((batch, 3), dtype=numpy.float32)
            actual = session.run(None, {"X": x})[0]
            numpy.testing.assert_array_equal(actual, x * 2 if batch > 3 else x - 2)

    def test_standalone_functions(self):
        import torch
        from yobx.torch import to_onnx, FunctionOptions

        model = torch.nn.Linear(4, 3).eval()
        x = torch.randn(2, 4)
        for promote in (False, True):
            with self.subTest(return_initializer=promote):
                artifact = to_onnx(
                    model,
                    (x,),
                    input_names=["X"],
                    as_function=True,
                    function_options=FunctionOptions(
                        name="Linear", domain="native.torch", return_initializer=promote
                    ),
                )
                self.assertIsInstance(artifact.proto, onnx.FunctionProto)
                weights = artifact.function.initializers_dict or {}
                caller = helper.make_model(
                    helper.make_graph(
                        [
                            helper.make_node(
                                "Linear", ["X", *weights], ["Y"], domain="native.torch"
                            )
                        ],
                        "caller",
                        [helper.make_tensor_value_info("X", 1, [2, 4])],
                        [helper.make_tensor_value_info("Y", 1, [2, 3])],
                        list(weights.values()),
                    ),
                    opset_imports=[
                        helper.make_opsetid("", 21),
                        helper.make_opsetid("native.torch", 1),
                    ],
                    ir_version=10,
                )
                caller.functions.append(artifact.proto)
                actual = self.session(caller).run(None, {"X": x.numpy()})[0]
                numpy.testing.assert_allclose(
                    actual, model(x).detach().numpy(), rtol=1e-5, atol=1e-6
                )

    def test_default_top_level_dispatch(self):
        import torch
        from yobx import to_onnx
        from yobx.builder.onnxlight import OnnxLightGraphBuilder

        model = torch.nn.Linear(3, 2).eval()
        x = torch.randn(4, 3)
        for backend in (None, "onnx-light"):
            with self.subTest(backend=backend):
                artifact = to_onnx(model, (x,), input_names=["X"], graph_backend=backend)
                self.assertIsInstance(artifact.builder, OnnxLightGraphBuilder)
                self.assertEqual(artifact.report.extra["torch_backend"], "native")
                actual = self.session(artifact.proto).run(None, {"X": x.numpy()})[0]
                numpy.testing.assert_allclose(
                    actual, model(x).detach().numpy(), rtol=1e-5, atol=1e-6
                )

    def test_mixed_precision_comparisons(self):
        """Preserves Torch operand promotion rather than narrowing to the first input."""
        import torch
        from yobx import to_onnx

        class Compare(torch.nn.Module):
            def forward(self, x, y):
                return x < y, x == y, x > y

        for left_dtype, right_dtype in (
            (torch.float32, torch.float64),
            (torch.float64, torch.float32),
        ):
            with self.subTest(left=left_dtype, right=right_dtype):
                x = torch.tensor([1.0, 2.0], dtype=left_dtype)
                y = torch.tensor([1.0 + 2**-40, 2.0], dtype=right_dtype)
                artifact = to_onnx(
                    Compare(), (x, y), input_names=["X", "Y"], graph_backend="onnx-light"
                )
                actual = self.session(artifact.proto).run(None, {"X": x.numpy(), "Y": y.numpy()})
                for result, expected in zip(actual, Compare()(x, y)):
                    numpy.testing.assert_array_equal(result, expected.numpy())

    def test_custom_dispatcher_and_tensor_initializers(self):
        import torch
        from yobx.builder.onnxlight import OnnxLightGraphBuilder
        from yobx.torch import to_onnx

        calls = []

        def relu(builder, state, outputs, x):
            self.assertIsInstance(builder, OnnxLightGraphBuilder)
            self.assertEqual(state["dtype"], torch.float32)
            calls.append(True)
            one = builder.make_initializer("", torch.tensor(1, dtype=torch.float32))
            return builder.op.Relu(builder.op.Mul(x, one), outputs=outputs)

        class Dispatcher:
            def find_function(self, target):
                return relu if target is torch.ops.aten.relu.default else None

        x = torch.randn(2, 3)
        artifact = to_onnx(torch.nn.ReLU(), (x,), input_names=["X"], dispatcher=Dispatcher())
        self.assertEqual(calls, [True])
        numpy.testing.assert_array_equal(
            self.session(artifact.proto).run(None, {"X": x.numpy()})[0], torch.relu(x).numpy()
        )
        with self.assertRaisesRegex(NotImplementedError, "concrete tensor storage"):
            OnnxLightGraphBuilder(18).make_initializer("meta", torch.empty(2, device="meta"))

    def test_parameter_names_and_repeated_outputs(self):
        import torch
        from yobx.torch import to_onnx

        model = torch.nn.Linear(4, 3)
        x = torch.randn(2, 4)
        artifact = to_onnx(model, (x,), input_names=["weight"])
        actual = self.session(artifact.proto).run(None, {"weight": x.numpy()})[0]
        numpy.testing.assert_allclose(actual, model(x).detach().numpy(), rtol=1e-5, atol=1e-6)

        class Repeat(torch.nn.Module):
            def forward(self, x):
                return x, x + 1, x

        artifact = to_onnx(Repeat(), (x,), input_names=["X"], validate_onnx=True)
        self.assertEqual(len(set(artifact.output_names)), 3)
        actual = self.session(artifact.proto).run(None, {"X": x.numpy()})
        for value, expected in zip(actual, Repeat()(x)):
            numpy.testing.assert_array_equal(value, expected.numpy())

    def test_convolution(self):
        import torch
        from yobx.torch import to_onnx

        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 3, padding=1), torch.nn.Sigmoid()
        ).eval()
        artifact = to_onnx(
            model, (torch.randn(2, 2, 5, 5),), input_names=["X"], dynamic_shapes=({0: "batch"},)
        )
        x = torch.randn(3, 2, 5, 5)
        actual = self.session(artifact.proto).run(None, {"X": x.numpy()})[0]
        numpy.testing.assert_allclose(actual, model(x).detach().numpy(), rtol=1e-5, atol=1e-6)

    def test_kwargs_scalar_promotion_and_validation(self):
        import torch
        from yobx.torch import to_onnx

        class Model(torch.nn.Module):
            def forward(self, x, *, y, scale=2):
                return x + scale * y

        x, y = torch.randn(2, 3), torch.randn(2, 3)
        artifact = to_onnx(
            Model(),
            (x,),
            kwargs={"y": y, "scale": 2},
            input_names=["X", "Y"],
            output_names=["result"],
            validate_onnx=True,
        )
        actual = self.session(artifact.proto).run(None, {"X": x.numpy(), "Y": y.numpy()})[0]
        numpy.testing.assert_allclose(actual, (x + 2 * y).numpy())

        class Promote(torch.nn.Module):
            def forward(self, x):
                return x + 0.5

        x = torch.arange(4, dtype=torch.int64)
        artifact = to_onnx(Promote(), (x,), input_names=["X"])
        actual = self.session(artifact.proto).run(None, {"X": x.numpy()})[0]
        numpy.testing.assert_array_equal(actual, Promote()(x).numpy())

    def test_exported_program_and_external_weights(self):
        import torch
        from yobx.container import ExportArtifact
        from yobx.torch import to_onnx

        model = torch.nn.Linear(4, 3).eval()
        x = torch.randn(2, 4)
        program = torch.export.export(model, (x,))
        artifact = to_onnx(program, input_names=["X"], large_model=True, external_threshold=0)
        self.assertIsNotNone(artifact.container)
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = os.path.join(directory, "torch.onnx")
            artifact.save(path)
            self.assertTrue(os.path.exists(os.path.join(directory, "torch.xlsx")))
            loaded = ExportArtifact.load(path)
            actual = self.session(loaded.get_proto()).run(None, {"X": x.numpy()})[0]
            numpy.testing.assert_allclose(actual, model(x).detach().numpy(), rtol=1e-5, atol=1e-6)

    def test_unsupported_requests_are_explicit(self):
        import torch
        from yobx.torch import to_onnx, ExportOptions

        x = torch.randn(2, 3)
        model = torch.nn.ReLU()
        with self.assertRaisesRegex(NotImplementedError, "submodule"):
            to_onnx(model, (x,), export_modules_as_functions=True)
        with self.assertRaisesRegex(NotImplementedError, "torch.export.export only"):
            to_onnx(model, (x,), export_options=ExportOptions(tracing=True))
        with self.assertRaisesRegex(TypeError, "OnnxLightOptimizationOptions"):
            to_onnx(model, (x,), options=object())
        with self.assertRaisesRegex(NotImplementedError, "flat arguments"):
            to_onnx(model, ((x,),))

        class Mutate(torch.nn.Module):
            def forward(self, x):
                return x.add_(1)

        with self.assertRaisesRegex(NotImplementedError, "mutations"):
            to_onnx(Mutate(), (x,))

        class Unsupported(torch.nn.Module):
            def forward(self, x):
                return torch.linalg.svd(x)

        with self.assertRaisesRegex(NotImplementedError, "no Python builder"):
            to_onnx(Unsupported(), (x,))

    def test_subprocess_blocks_legacy_engines(self):
        code = textwrap.dedent("""
            import importlib.abc
            import importlib.machinery
            import sys
            blocked = (
                "onnx", "yobx.xoptim", "yobx.torch.interpreter.interpreter",
                "yobx.xshape.shape_type_compute",
            )
            class RejectLegacyEngines(importlib.abc.MetaPathFinder, importlib.abc.Loader):
                def find_spec(self, fullname, path=None, target=None):
                    # Dynamo probes optional package specs without importing them.
                    if fullname == "onnx":
                        return importlib.machinery.ModuleSpec(fullname, self)
                    if any(fullname == name or fullname.startswith(name + ".")
                           for name in blocked):
                        raise AssertionError("Legacy engine import: " + fullname)
                    return None
                def create_module(self, spec):
                    raise AssertionError("Reference ONNX import: " + spec.name)
                def exec_module(self, module):
                    raise AssertionError("Reference ONNX execution")
            sys.meta_path.insert(0, RejectLegacyEngines())
            import torch
            from onnx_light import onnx
            from yobx.torch import to_onnx
            model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
            artifact = to_onnx(
                model, (torch.randn(2, 4),), dynamic_shapes=({0: "batch"},),
                return_optimize_report=True,
            )
            assert isinstance(artifact.proto, onnx.ModelProto)
            assert artifact.report.extra["backend"] == "onnx-light"
            assert artifact.report.extra["rewrites"] > 0
            assert not any(
                module == name or module.startswith(name + ".")
                for name in blocked for module in sys.modules
            )
            print("native Torch engines verified")
            """)
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native Torch engines verified", result.stdout)


if __name__ == "__main__":
    unittest.main()
