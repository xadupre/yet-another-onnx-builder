"""Tests the converter bridge against the published native onnx-light wheel."""

import importlib.util
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import numpy
from onnx_light import onnx
from onnx_light.onnx import checker
from onnx_light.onnx.reference import ReferenceEvaluator


@unittest.skipUnless(importlib.util.find_spec("onnx_light"), "onnx-light wheel is not installed")
class TestOnnxLightGraphBuilder(unittest.TestCase):
    def make_builder(self, *args, **kwargs):
        from yobx.builder.onnxlight import OnnxLightGraphBuilder

        return OnnxLightGraphBuilder(*args, **kwargs)

    def test_converter_metadata_and_scalar_initializers(self):
        builder = self.make_builder(18)
        builder.make_tensor_input("X", onnx.TensorProto.FLOAT, ("batch", 4), device=-1)
        self.assertEqual(builder.unique_dimension_name("batch"), "batch_2")
        self.assertEqual(builder.unique_dimension_name("batch"), "batch_3")
        builder.set_type_shape_or_rank("Y", "X")
        self.assertEqual(builder.get_shape("Y"), ("batch", 4))
        self.assertEqual(builder.get_device("Y"), -1)
        name = builder.make_initializer("weight", numpy.int64(4), parameter_name="layer.weight")
        self.assertEqual(name, "layer.weight")
        self.assertEqual(builder.get_shape(name), ())
        self.assertEqual(builder.get_constant(name).dtype, numpy.dtype("int64"))
        with self.assertRaises(TypeError):
            builder.set_device("Y", "cuda")

    def test_native_function_export_constants_and_parameters(self):
        from yobx.xbuilder.function_options import FunctionOptions

        builder = self.make_builder(18, as_function=True)
        builder.make_tensor_input("X", onnx.TensorProto.FLOAT, (2, 4))
        weights = numpy.arange(12, dtype=numpy.float32).reshape(4, 3)
        builder.op.MatMul("X", weights, outputs=["Y"])
        builder.make_tensor_output("Y")
        feeds = {"X": numpy.arange(8, dtype=numpy.float32).reshape(2, 4)}
        for return_initializer in (False, True):
            with self.subTest(return_initializer=return_initializer):
                artifact = builder.to_onnx(
                    optimize=False,
                    function_options=FunctionOptions(
                        name="Linear", domain="custom", return_initializer=return_initializer
                    ),
                )
                self.assertIsInstance(artifact.proto, onnx.FunctionProto)
                self.assertEqual(str(artifact.proto.name), "Linear")
                values = {**feeds, **artifact.function.initializers_dict}
                actual = ReferenceEvaluator(artifact.proto).run(None, values)[0]
                numpy.testing.assert_array_equal(actual, feeds["X"] @ weights)
                self.assertEqual(len(artifact.proto.input), 2 if return_initializer else 1)
        self.assertEqual(len(builder.initializers_dict), 1)

    def test_native_local_function_converter_registration(self):
        from yobx.xbuilder.function_options import FunctionOptions

        function = self.make_builder(18, as_function=True)
        function.make_tensor_input("X", onnx.TensorProto.FLOAT, (2, 4))
        weights = numpy.ones((4, 3), dtype=numpy.float32)
        function.op.MatMul("X", weights, outputs=["Y"])
        function.make_tensor_output("Y")
        builder = self.make_builder(18)
        builder.make_tensor_input("A", onnx.TensorProto.FLOAT, (2, 4))
        initializers, key = builder.make_local_function(
            function,
            FunctionOptions(name="Linear", domain="custom", move_initializer_to_constant=True),
        )
        self.assertEqual(initializers, [])
        self.assertEqual(key, ("custom", "Linear"))
        builder.make_node(key[1], ["A"], ["B"], domain=key[0])
        self.assertEqual(builder.get_shape("B"), (2, 3))
        builder.make_tensor_output("B")
        feeds = {"A": numpy.ones((2, 4), dtype=numpy.float32)}
        for inline in (False, True):
            with self.subTest(inline=inline):
                model = builder.to_native(optimize=False, inline=inline)
                self.assertEqual(len(model.functions), 0 if inline else 1)
                actual = ReferenceEvaluator(model).run(None, feeds)[0]
                numpy.testing.assert_array_equal(actual, feeds["A"] @ weights)

    def test_native_local_function_merge_and_rename(self):
        from yobx.xbuilder.function_options import FunctionOptions

        function = self.make_builder(18, as_function=True)
        function.make_tensor_input("X", onnx.TensorProto.FLOAT, (2, 3))
        function.op.Add("X", numpy.ones(3, dtype=numpy.float32), outputs=["Y"])
        function.make_tensor_output("Y")
        builder = self.make_builder(18)
        options = FunctionOptions(
            name="AddBias", domain="custom", return_initializer=True, merge_allowed=True
        )
        first = builder.make_local_function(function, options)
        second = builder.make_local_function(function, options)
        self.assertEqual(first, second)
        self.assertEqual(len(builder.functions), 1)
        self.assertEqual(len(builder.initializers_dict), 1)
        with self.assertRaisesRegex(ValueError, "already exists"):
            builder.make_local_function(
                function, FunctionOptions(name="AddBias", domain="custom")
            )
        _, renamed = builder.make_local_function(
            function, FunctionOptions(name="AddBias", domain="custom", rename_allowed=True)
        )
        self.assertEqual(renamed, ("custom", "AddBias_2"))
        self.assertEqual(len(builder.functions), 2)

    def test_native_nested_functions_and_imported_annotations(self):
        from yobx.xbuilder.function_options import FunctionOptions

        leaf = self.make_builder(18, as_function=True)
        leaf.make_tensor_input("X", onnx.TensorProto.FLOAT, ("batch", 3))
        leaf.op.Add("X", numpy.ones(3, dtype=numpy.float32), outputs=["Y"])
        leaf.make_tensor_output("Y")
        wrapper = self.make_builder(18, as_function=True)
        wrapper.make_tensor_input("X", onnx.TensorProto.FLOAT, ("batch", 3))
        wrapper.make_local_function(leaf, FunctionOptions(name="Bias", domain="custom"))
        wrapper.op.Neg(wrapper.op.Bias("X", domain="custom"), outputs=["Y"])
        wrapper.make_tensor_output("Y")

        builder = self.make_builder(18)
        builder.make_tensor_input("X", onnx.TensorProto.FLOAT, ("batch", 3))
        options = FunctionOptions(name="NegativeBias", domain="custom", merge_allowed=True)
        first = builder.make_local_function(wrapper, options)
        self.assertEqual(builder.make_local_function(wrapper, options), first)
        self.assertEqual(len(builder.functions), 2)
        builder.op.Neg(builder.op.NegativeBias("X", domain="custom"), outputs=["Y"])
        builder.make_tensor_output("Y")
        self.assertEqual(builder.get_shape("Y"), ("batch", 3))

        imported = self.make_builder(builder.to_native(optimize=False, inline=False))
        imported.set_shape("X", (5, 3))
        imported.op.Bias("Y", domain="custom", outputs=["Z"])
        imported.make_tensor_output("Z")
        self.assertEqual(imported.get_shape("Z"), (5, 3))
        subset = imported.make_subset_builder(["X"], add_local_functions=True)
        subset.op.NegativeBias("X", domain="custom", outputs=["Z"])
        subset.make_tensor_output("Z")
        feeds = {"X": numpy.arange(15, dtype=numpy.float32).reshape(5, 3)}
        for inline in (False, True):
            for optimize in (False, True):
                with self.subTest(inline=inline, optimize=optimize):
                    model = imported.to_native(optimize=optimize, inline=inline)
                    checker.check_model(model)
                    self.assertEqual(len(model.functions), 0 if inline else 2)
                    actual = ReferenceEvaluator(model).run(None, feeds)
                    numpy.testing.assert_array_equal(actual[0], feeds["X"] + 1)
                    numpy.testing.assert_array_equal(actual[1], feeds["X"] + 2)
                    model = subset.to_native(optimize=optimize, inline=inline)
                    checker.check_model(model)
                    numpy.testing.assert_array_equal(
                        ReferenceEvaluator(model).run(None, feeds)[0], -(feeds["X"] + 1)
                    )

    def test_native_imported_function_referenced_attribute(self):
        from onnx_light.onnx import helper

        attribute = onnx.AttributeProto()
        attribute.name = "alpha"
        attribute.ref_attr_name = "slope"
        attribute.type = onnx.AttributeProto.FLOAT
        node = helper.make_node("LeakyRelu", ["X"], ["Y"])
        node.attribute.append(attribute)
        function = helper.make_function(
            "custom",
            "Activation",
            ["X"],
            ["Y"],
            [node],
            [helper.make_opsetid("", 18)],
            attributes=["slope"],
        )
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Activation", ["X"], ["activation"], domain="custom", slope=0.25
                    ),
                    helper.make_node("Neg", ["activation"], ["Y"]),
                ],
                "referenced_attribute",
                [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3])],
                [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 3])],
            ),
            functions=[function],
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom", 1)],
        )
        builder = self.make_builder(model)
        builder.op.Activation("Y", domain="custom", slope=0.5, outputs=["Z"])
        builder.make_tensor_output("Z")
        feeds = {"X": numpy.arange(-3, 3, dtype=numpy.float32).reshape(2, 3)}
        expected = -numpy.where(feeds["X"] < 0, feeds["X"] * 0.25, feeds["X"])
        for inline in (False, True):
            with self.subTest(inline=inline):
                exported = builder.to_native(inline=inline)
                checker.check_model(exported)
                actual = ReferenceEvaluator(exported).run(None, feeds)
                numpy.testing.assert_array_equal(actual[0], expected)
                numpy.testing.assert_array_equal(
                    actual[1], numpy.where(expected < 0, expected * 0.5, expected)
                )

    def test_native_nested_function_merge_rejects_conflicting_body(self):
        from yobx.xbuilder.function_options import FunctionOptions

        def make_wrapper(bias):
            leaf = self.make_builder(18, as_function=True)
            leaf.make_tensor_input("X", onnx.TensorProto.FLOAT, (2, 3))
            leaf.op.Add("X", numpy.full(3, bias, dtype=numpy.float32), outputs=["Y"])
            leaf.make_tensor_output("Y")
            wrapper = self.make_builder(18, as_function=True)
            wrapper.make_tensor_input("X", onnx.TensorProto.FLOAT, (2, 3))
            wrapper.make_local_function(leaf, FunctionOptions(name="Bias", domain="custom"))
            wrapper.op.Bias("X", domain="custom", outputs=["Y"])
            wrapper.make_tensor_output("Y")
            return wrapper

        builder = self.make_builder(18)
        options = FunctionOptions(name="Wrapper", domain="custom", merge_allowed=True)
        builder.make_local_function(make_wrapper(1), options)
        before = builder.to_native(optimize=False, inline=False).SerializeToString()
        with self.assertRaisesRegex(ValueError, "Conflicting nested local function"):
            builder.make_local_function(make_wrapper(2), options)
        self.assertEqual(
            builder.to_native(optimize=False, inline=False).SerializeToString(), before
        )

    def test_native_cleanup_removes_unused_initializers(self):
        builder = self.make_builder(18)
        builder.make_tensor_input("X", onnx.TensorProto.FLOAT, (2, 3))
        builder.make_initializer("unused", numpy.ones(3, dtype=numpy.float32))
        builder.op.Add("X", numpy.ones(3, dtype=numpy.float32), outputs=["Y"])
        builder.make_tensor_output("Y")
        builder.remove_unused()
        self.assertNotIn("unused", builder.initializers_dict)
        exported = builder.to_native()
        checker.check_model(exported)
        feeds = {"X": numpy.zeros((2, 3), dtype=numpy.float32)}
        numpy.testing.assert_array_equal(
            ReferenceEvaluator(exported).run(None, feeds)[0], feeds["X"] + 1
        )

    def test_native_export_tracks_captured_subgraph_values(self):
        from onnx_light.onnx import helper

        branch = helper.make_graph(
            [
                helper.make_node("Add", ["X", "bias"], ["sum"]),
                helper.make_node("Neg", ["sum"], ["branch_output"]),
            ],
            "branch",
            [],
            [helper.make_tensor_value_info("branch_output", onnx.TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Constant",
                        [],
                        ["bias"],
                        value=onnx.numpy_helper.from_array(
                            numpy.array([2, 3], dtype=numpy.float32)
                        ),
                    ),
                    helper.make_node(
                        "If", ["condition"], ["Y"], then_branch=branch, else_branch=branch
                    ),
                    helper.make_node("Identity", ["X"], ["unused"]),
                ],
                "captures",
                [
                    helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2]),
                    helper.make_tensor_value_info("condition", onnx.TensorProto.BOOL, []),
                ],
                [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        builder = self.make_builder(model)
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                exported = builder.to_native(optimize=optimize)
                checker.check_model(exported)
                for condition in (False, True):
                    result = ReferenceEvaluator(exported).run(
                        None,
                        {
                            "X": numpy.array([4, 5], dtype=numpy.float32),
                            "condition": numpy.array(condition),
                        },
                    )[0]
                    numpy.testing.assert_array_equal(
                        result, numpy.array([-6, -8], dtype=numpy.float32)
                    )

    def test_native_export_tracks_forwarded_subgraph_outputs(self):
        from onnx_light.onnx import helper

        branch = helper.make_graph(
            [helper.make_node("Identity", ["bias"], ["forwarded"])],
            "forward",
            [],
            [helper.make_tensor_value_info("forwarded", onnx.TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Identity", ["X"], ["bias"]),
                    helper.make_node(
                        "If", ["condition"], ["Y"], then_branch=branch, else_branch=branch
                    ),
                ],
                "forwarded_capture",
                [
                    helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1]),
                    helper.make_tensor_value_info("condition", onnx.TensorProto.BOOL, []),
                ],
                [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                exported = self.make_builder(model).to_native(optimize=optimize)
                checker.check_model(exported)
                feeds = {"X": numpy.array([3], dtype=numpy.float32)}
                for condition in (False, True):
                    result = ReferenceEvaluator(exported).run(
                        None, {**feeds, "condition": numpy.array(condition)}
                    )[0]
                    numpy.testing.assert_array_equal(result, feeds["X"])

    def test_native_import_rejects_invalid_dependencies(self):
        from onnx_light.onnx import helper

        model = helper.make_model(
            helper.make_graph([], "empty", [], []), opset_imports=[helper.make_opsetid("", 18)]
        )
        model.graph.node.extend(
            [
                helper.make_node("Identity", ["B"], ["A"]),
                helper.make_node("Identity", ["A"], ["B"]),
            ]
        )
        with self.assertRaises(ValueError):
            self.make_builder(model)
        model.graph.ClearField("node")
        model.graph.node.append(helper.make_node("Identity", ["missing"], ["A"]))
        with self.assertRaisesRegex(ValueError, "missing"):
            self.make_builder(model)

    def test_dimension_materialization_uses_native_shape_operators(self):
        builder = self.make_builder(18)
        builder.make_tensor_input("X", onnx.TensorProto.FLOAT, ("batch", 3))
        name = builder.get_dimension_as_result("batch")
        self.assertEqual(builder.get_shape(name), ())
        self.assertEqual(builder.value_as_shape(name), ("batch",))
        builder.make_tensor_output(name)
        model = builder.to_native(optimize=False)
        actual = ReferenceEvaluator(model).run(
            None, {"X": numpy.ones((5, 3), dtype=numpy.float32)}
        )[0]
        self.assertEqual(actual, 5)
        with self.assertRaisesRegex(ValueError, "No input shape"):
            builder.get_dimension_as_result("missing")

    def test_native_export_without_reference_imports(self):
        code = textwrap.dedent("""
            import importlib.abc
            import os
            import sys
            import tempfile

            blocked = (
                "onnx", "yobx.xbuilder", "yobx.reference"
            )

            class RejectReferenceImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if any(fullname == name or fullname.startswith(name + ".")
                           for name in blocked):
                        raise AssertionError("Non-native import attempted: " + fullname)
                    return None

            sys.meta_path.insert(0, RejectReferenceImports())

            from onnx_light import onnx
            from yobx.builder.onnxlight import (
                OnnxLightGraphBuilder, OnnxLightOptimizationOptions
            )

            options = OnnxLightOptimizationOptions(["TransposeTranspose"])
            builder = OnnxLightGraphBuilder(18, ir_version=8, optimization_options=options)
            builder.make_tensor_input("X", onnx.TensorProto.FLOAT, ("N", 3))
            builder.set_shape("X", ("batch", 3))
            transposed = builder.op.Transpose("X", perm=[1, 0])
            result = builder.op.Transpose(transposed, perm=[1, 0], outputs=["Y"])
            assert builder.get_shape(result) == ("batch", 3)
            assert builder.shapes_context.get(result).dtype == onnx.TensorProto.FLOAT
            builder.make_tensor_output(result)
            native = builder.to_native()
            assert isinstance(native, onnx.ModelProto)
            assert [str(node.op_type) for node in native.graph.node] == ["Identity"]
            assert native.ir_version == 8
            assert [str(opset.domain) for opset in native.opset_import] == [""]
            assert all(str(node.domain) == "" for node in native.graph.node)
            assert len(builder.to_native(optimize=False).graph.node) == 2
            artifact = builder.to_onnx(return_optimize_report=True)
            assert isinstance(artifact.proto, onnx.ModelProto)
            assert artifact.report.extra["rewrites"] == 1
            assert len(artifact.proto.graph.node) == 1
            from yobx.container import ExportArtifact
            with tempfile.TemporaryDirectory(dir=".") as directory:
                path = os.path.join(directory, "native.onnx")
                artifact.save(path)
                assert os.path.exists(os.path.join(directory, "native.xlsx"))
                reloaded = ExportArtifact.load(path)
                assert isinstance(reloaded.proto, onnx.ModelProto)
                assert len(reloaded.proto.graph.node) == 1

            original = builder.to_native(optimize=False)
            original.producer_name = "native-producer"
            original.producer_version = "sentinel-version"
            original.domain = "sentinel-domain"
            original.model_version = 42
            original.doc_string = "model documentation"
            original.graph.name = "original graph"
            original.graph.doc_string = "graph documentation"
            original.metadata_props.add(key="model-key", value="model-value")
            original.graph.metadata_props.add(key="graph-key", value="graph-value")
            original_bytes = original.SerializeToString()
            imported = OnnxLightGraphBuilder(original, optimization_options=options)
            restored = imported.to_native()
            assert [str(node.op_type) for node in restored.graph.node] == ["Identity"]
            assert restored.ir_version == original.ir_version == 8
            for field in ("producer_name", "producer_version", "domain",
                          "model_version", "doc_string"):
                assert getattr(restored, field) == getattr(original, field), field
            assert restored.graph.name == original.graph.name
            assert restored.graph.doc_string == original.graph.doc_string
            assert {str(p.key): str(p.value) for p in restored.metadata_props} == {
                "model-key": "model-value"
            }
            assert {str(p.key): str(p.value) for p in restored.graph.metadata_props} == {
                "graph-key": "graph-value"
            }
            assert original.SerializeToString() == original_bytes
            assert len(imported.to_native(optimize=False).graph.node) == 2
            reparsed = onnx.ModelProto()
            reparsed.ParseFromString(restored.SerializeToString())
            assert len(reparsed.graph.node) == 1
            overridden = OnnxLightGraphBuilder(original, ir_version=10)
            assert overridden.to_native(optimize=False).ir_version == 10

            default_ir = OnnxLightGraphBuilder(18)
            default_ir.make_tensor_input("X", onnx.TensorProto.FLOAT, [3])
            default_ir.make_tensor_output("X")
            assert default_ir.to_native(optimize=False).ir_version == (
                default_ir.inner_builder.to_onnx().ir_version
            )
            assert not any(
                module == name or module.startswith(name + ".")
                for module in sys.modules for name in blocked
            )
            print("native-only export and metadata roundtrip passed")
            """)
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native-only export and metadata roundtrip passed", result.stdout)

    def test_native_constant_folding_without_evaluator_import(self):
        code = textwrap.dedent("""
            import importlib.abc
            import sys
            import numpy

            class RejectEvaluatorImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    blocked = (
                        "onnx", "yobx.reference", "onnx_light.onnx.reference",
                        "onnx_light._reference",
                    )
                    if any(fullname == name or fullname.startswith(name + ".")
                           for name in blocked):
                        raise AssertionError("Evaluator import attempted: " + fullname)
                    return None

            sys.meta_path.insert(0, RejectEvaluatorImports())
            from yobx.builder.onnxlight import OnnxLightGraphBuilder
            from onnx_light.onnx import checker

            builder = OnnxLightGraphBuilder(18)
            builder.make_tensor_input("X", 1, (2, 3))
            shape = builder.op.Concat(
                numpy.array([2], dtype=numpy.int64),
                numpy.array([3], dtype=numpy.int64),
                axis=0,
            )
            builder.op.Reshape("X", shape, outputs=["Y"])
            builder.make_tensor_output("Y")
            model = builder.to_native()
            checker.check_model(model)
            assert all(str(node.op_type) != "Concat" for node in model.graph.node)
            assert builder.get_shape("Y") == (2, 3)
            print("native Concat folding without evaluator passed")
            """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONPATH": os.getcwd()},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native Concat folding without evaluator passed", result.stdout)

    def test_dynamic_add_native_engines(self):
        from onnx_light.onnx_core.graph_builder import GraphBuilder
        from onnx_light.onnx_core.shape_inference import ShapesContext
        from yobx.typing import GraphBuilderExtendedProtocol

        builder = self.make_builder(18)
        self.assertIsInstance(builder.inner_builder, GraphBuilder)
        self.assertIsInstance(builder.shapes_context, ShapesContext)
        self.assertIsInstance(builder, GraphBuilderExtendedProtocol)
        builder.make_tensor_input("X", onnx.TensorProto.FLOAT, ("batch", 3))
        builder.make_tensor_input("Y", onnx.TensorProto.FLOAT, ("batch", 3))
        output = builder.op.Add("X", "Y", outputs="Z")
        self.assertEqual(output, "Z")
        self.assertEqual(builder.get_shape(output), ("batch", 3))
        self.assertEqual(builder.get_type(output), onnx.TensorProto.FLOAT)
        builder.make_tensor_output(output)
        artifact = builder.to_onnx(optimize=False, return_optimize_report=True)
        checker.check_model(artifact.proto)
        self.assertEqual(artifact.proto.ir_version, 8)
        self.assertEqual(artifact.report.extra["backend"], "onnx-light")
        self.assertEqual(builder.input_names, ["X", "Y"])
        self.assertEqual(builder.output_names, ["Z"])
        for batch in (1, 5):
            x = numpy.arange(batch * 3, dtype=numpy.float32).reshape(batch, 3)
            actual = ReferenceEvaluator(artifact.proto).run(None, {"X": x, "Y": x + 1})[0]
            numpy.testing.assert_array_equal(actual, x + x + 1)

    def test_native_pattern_report_and_import(self):
        from yobx.builder.onnxlight import OnnxLightOptimizationOptions

        source = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node("Transpose", ["X"], ["T"], perm=[1, 0]),
                    onnx.helper.make_node("Transpose", ["T"], ["Y"], perm=[1, 0]),
                ],
                "original_graph",
                [onnx.helper.make_tensor_value_info("X", 1, ["N", 3])],
                [onnx.helper.make_tensor_value_info("Y", 1, ["N", 3])],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=8,
            producer_name="metadata-producer",
            producer_version="version",
        )
        source.domain = "example"
        source.model_version = 42
        source.doc_string = "model documentation"
        source.graph.doc_string = "graph documentation"
        source.graph.metadata_props.add(key="graph-key", value="graph-value")
        source.metadata_props.add(key="custom", value="preserved")
        original_bytes = source.SerializeToString()
        builder = self.make_builder(
            source, optimization_options=OnnxLightOptimizationOptions(["TransposeTranspose"])
        )
        unoptimized = builder.to_onnx(optimize=False).proto
        self.assertEqual(len(unoptimized.graph.node), 2)
        artifact = builder.to_onnx(return_optimize_report=True)
        self.assertEqual([node.op_type for node in artifact.proto.graph.node], ["Identity"])
        self.assertEqual(artifact.report.extra["rewrites"], 1)
        self.assertEqual(artifact.report.stats[0]["pattern"], "TransposeTranspose")
        self.assertEqual(artifact.report.stats[0]["added"], 1)
        self.assertEqual(artifact.report.stats[0]["removed"], 2)
        self.assertEqual(artifact.proto.producer_name, source.producer_name)
        self.assertEqual(artifact.proto.producer_version, source.producer_version)
        self.assertEqual(artifact.proto.domain, source.domain)
        self.assertEqual(artifact.proto.model_version, source.model_version)
        self.assertEqual(artifact.proto.ir_version, source.ir_version)
        self.assertEqual(artifact.proto.doc_string, source.doc_string)
        self.assertEqual(artifact.proto.graph.doc_string, source.graph.doc_string)
        self.assertEqual(artifact.proto.graph.name, source.graph.name)
        self.assertEqual(
            [(p.key, p.value) for p in artifact.proto.graph.metadata_props],
            [(p.key, p.value) for p in source.graph.metadata_props],
        )
        self.assertEqual(
            [(p.key, p.value) for p in artifact.proto.metadata_props],
            [(p.key, p.value) for p in source.metadata_props],
        )
        self.assertEqual(source.SerializeToString(), original_bytes)
        self.assertEqual(len(builder.to_onnx(optimize=False).proto.graph.node), 2)
        checker.check_model(artifact.proto)

    def test_empty_patterns_are_really_empty(self):
        from yobx.builder.onnxlight import OnnxLightOptimizationOptions

        builder = self.make_builder(18, optimization_options=OnnxLightOptimizationOptions([]))
        builder.make_tensor_input("X", 1, ("N", 3))
        transposed = builder.op.Transpose("X", perm=[1, 0])
        output = builder.op.Transpose(transposed, perm=[1, 0])
        builder.make_tensor_output(output)
        artifact = builder.to_onnx(return_optimize_report=True)
        self.assertEqual(artifact.report.extra["rewrites"], 0)
        self.assertEqual([node.op_type for node in artifact.proto.graph.node], ["Transpose"] * 2)

    def test_shape_type_overrides_and_rank(self):
        builder = self.make_builder(18)
        builder.make_tensor_input("X", 1, ("N", 3))
        builder.set_type("X", 11)
        builder.set_shape("X", ("batch", 4))
        self.assertEqual(builder.shapes_context.get("X").dtype, 11)
        self.assertEqual(builder.inner_builder.get_shape("X").dtype, 11)
        self.assertEqual(builder.inner_builder.get_shape("X").shape.dims(), ["batch", 4])
        output = builder.op.Identity("X")
        self.assertEqual(builder.get_type(output), 11)
        self.assertEqual(builder.get_shape(output), ("batch", 4))
        builder.make_tensor_output(output)
        artifact = builder.to_onnx(optimize=False)
        checker.check_model(artifact.proto)
        self.assertEqual(artifact.proto.graph.output[0].type.tensor_type.elem_type, 11)
        builder.set_rank("rank_only", 3)
        builder.set_type("rank_only", 1)
        self.assertTrue(builder.has_rank("rank_only"))
        self.assertEqual(builder.get_rank("rank_only"), 3)
        self.assertEqual(builder.get_shape("rank_only"), (None, None, None))
        with self.assertRaises(ValueError):
            builder.set_rank("rank_only", 2)
        with self.assertRaises(ValueError):
            builder.set_shape("rank_only", (0, 3))

    def test_initializer_names_constants_and_shapes(self):
        builder = self.make_builder(18)
        values = numpy.array([2, 3], dtype=numpy.int64)
        first = builder.make_initializer("shape", values)
        second = builder.make_initializer("shape", values)
        self.assertNotEqual(first, second)
        self.assertTrue(builder.is_constant(first))
        numpy.testing.assert_array_equal(builder.get_constant(first), values)
        self.assertEqual(builder.value_as_shape(first), (2, 3))
        self.assertEqual(builder.get_constant(first, as_shape=True), (2, 3))
        self.assertIsNone(builder.get_constant("missing", exc=False))
        with self.assertRaises(ValueError):
            builder.make_initializer("shape", values, give_unique_name=False)
        with self.assertRaises(TypeError):
            builder.make_initializer("bad", object())
        reference_tensor = onnx.numpy_helper.from_array(numpy.ones((2, 3), dtype=numpy.float32))
        name = builder.make_initializer("reference", reference_tensor)
        builder.make_tensor_output(name)
        checker.check_model(builder.to_onnx(optimize=False).proto)
        with builder.prefix_name_context("step"), builder.prefix_name_context("nested"):
            scoped = builder.unique_name("shape")
        self.assertEqual(scoped, "step__nested__shape")
        self.assertEqual(builder.unique_name("fresh"), "fresh")

    def test_requested_node_names_are_unique(self):
        builder = self.make_builder(18, ir_version=8)
        builder.make_tensor_input("X", 1, ("batch", 3))
        first = builder.op.Add("X", numpy.array(1, dtype=numpy.float32), name="main")
        second = builder.op.Mul(first, numpy.array(2, dtype=numpy.float32), name="main")
        output = builder.op.Relu(second, name="main_2")
        builder.make_tensor_output(output)
        names = [str(node.name) for node in builder.nodes]
        self.assertEqual(names, ["main", "main_2", "main_2_2"])
        self.assertEqual(builder.unique_name("main"), "main_3")

        restored = self.make_builder(builder.to_native(optimize=False))
        identity = restored.op.Identity(output, name="main")
        restored.make_tensor_output(identity)
        model = restored.to_onnx(optimize=False).proto
        self.assertEqual([str(node.name) for node in model.graph.node], [*names, "main_3"])
        checker.check_model(model)

        from onnxruntime import InferenceSession

        session = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        x = numpy.arange(6, dtype=numpy.float32).reshape(2, 3) - 2
        for actual in session.run(None, {"X": x}):
            numpy.testing.assert_array_equal(actual, numpy.maximum((x + 1) * 2, 0))

    def test_native_optimizer_node_names_are_unique(self):
        from onnxruntime import InferenceSession
        from yobx.builder.onnxlight import OnnxLightOptimizationOptions

        builder = self.make_builder(
            18, optimization_options=OnnxLightOptimizationOptions(["ReduceArgTopK"])
        )
        builder.make_tensor_input("X", 1, ("N", 3))
        maximum = builder.op.ReduceMax(
            "X", numpy.array([1], dtype=numpy.int64), keepdims=0, name="reduce"
        )
        index = builder.op.ArgMax("X", axis=1, keepdims=0, name="join_ridx")
        builder.make_tensor_output([maximum, index])
        artifact = builder.to_onnx(return_optimize_report=True)
        self.assertTrue(any(row["pattern"] == "ReduceArgTopK" for row in artifact.report.stats))
        names = [str(node.name) for node in artifact.proto.graph.node if node.name]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("ReduceArgTopKPattern--join_ridx", names)
        self.assertIn("ReduceArgTopKPattern--join_ridx_2", names)
        native = builder.to_native()
        self.assertEqual([str(node.name) for node in native.graph.node if node.name], names)
        session = InferenceSession(
            artifact.proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        x = numpy.array([[1, 3, 2], [9, 0, 1]], dtype=numpy.float32)
        maximum, index = session.run(None, {"X": x})
        numpy.testing.assert_array_equal(maximum, x.max(axis=1))
        numpy.testing.assert_array_equal(index, x.argmax(axis=1))

    def test_export_normalizes_all_node_name_scopes(self):
        from onnxruntime import InferenceSession

        nodes = [
            onnx.helper.make_node("Identity", ["captured"], ["A"], name="repeat"),
            onnx.helper.make_node("Relu", ["A"], ["B"], name="repeat"),
            onnx.helper.make_node("Identity", ["B"], ["Y"], name="repeat_2"),
        ]
        branch = onnx.helper.make_graph(
            nodes, "branch", [], [onnx.helper.make_tensor_value_info("Y", 1, ["N", 3])]
        )
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node("Identity", ["X"], ["outer"], name="if"),
                    onnx.helper.make_node("Identity", ["outer"], ["captured"], name="if"),
                    onnx.helper.make_node(
                        "If",
                        ["condition"],
                        ["result"],
                        name="if_2",
                        then_branch=branch,
                        else_branch=branch,
                    ),
                ],
                "main",
                [
                    onnx.helper.make_tensor_value_info("X", 1, ["N", 3]),
                    onnx.helper.make_tensor_value_info("condition", 9, []),
                ],
                [onnx.helper.make_tensor_value_info("result", 1, ["N", 3])],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=8,
        )
        model.functions.append(
            onnx.helper.make_function(
                "native.test",
                "unused",
                ["captured"],
                ["Y"],
                nodes,
                [onnx.helper.make_opsetid("", 18)],
            )
        )
        original = model.SerializeToString()
        builder = self.make_builder(model)
        normalized = builder.to_native(optimize=False, inline=False)
        self.assertEqual(
            [str(node.name) for node in normalized.graph.node], ["if", "if_3", "if_2"]
        )
        for attribute in normalized.graph.node[2].attribute:
            self.assertEqual(
                [str(node.name) for node in attribute.g.node], ["repeat", "repeat_3", "repeat_2"]
            )
        self.assertEqual(
            [str(node.name) for node in normalized.functions[0].node],
            ["repeat", "repeat_3", "repeat_2"],
        )
        self.assertEqual(model.SerializeToString(), original)
        session = InferenceSession(
            builder.to_onnx(optimize=False, inline=False).proto.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        x = numpy.array([[-1, 2, 3]], dtype=numpy.float32)
        actual = session.run(None, {"X": x, "condition": numpy.array(True)})[0]
        numpy.testing.assert_array_equal(actual, numpy.maximum(x, 0))

    @unittest.skipUnless(
        "scikit-learn" in importlib.metadata.packages_distributions().get("sklearn", []),
        "scikit-learn is not installed",
    )
    def test_sklearn_converter_repeated_node_names(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from sklearn.tree import DecisionTreeRegressor
        from onnxruntime import InferenceSession
        from yobx.sklearn import to_onnx
        from yobx.builder.onnxlight import OnnxLightGraphBuilder

        x = numpy.array([[1, 2, 3], [2, 4, 9]], dtype=numpy.float32)
        y = numpy.array([1, 2], dtype=numpy.float32)
        estimators = [
            MinMaxScaler(),
            make_pipeline(StandardScaler(), Ridge()),
            make_pipeline(StandardScaler(), DecisionTreeRegressor(random_state=0)),
        ]
        for estimator in estimators:
            with self.subTest(estimator=type(estimator).__name__):
                estimator.fit(x, y)
                artifact = to_onnx(estimator, (x,), builder_cls=OnnxLightGraphBuilder)
                names = [str(node.name) for node in artifact.proto.graph.node if node.name]
                self.assertEqual(len(names), len(set(names)))
                session = InferenceSession(
                    artifact.proto.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                expected = (
                    estimator.predict(x)
                    if hasattr(estimator, "predict")
                    else estimator.transform(x)
                )
                actual = session.run(None, {"X": x})[0].reshape(expected.shape)
                numpy.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_any_opset_helpers_and_multi_outputs(self):
        for opset in (11, 13, 18):
            with self.subTest(opset=opset):
                builder = self.make_builder(opset)
                builder.make_tensor_input("X", 1, ("N", 3))
                reduced = builder.op.ReduceMeanAnyOpset(
                    "X", numpy.array([1], dtype=numpy.int64), keepdims=1
                )
                squeezed = builder.op.SqueezeAnyOpset(
                    reduced, numpy.array([1], dtype=numpy.int64), outputs=["Y"]
                )
                builder.make_tensor_output(squeezed)
                model = builder.to_onnx(optimize=False).proto
                checker.check_model(model)
                x = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
                actual = ReferenceEvaluator(model).run(None, {"X": x})[0]
                numpy.testing.assert_allclose(actual, x.mean(axis=1))
        builder = self.make_builder(18)
        builder.make_tensor_input("X", 1, ("N", 3))
        values, indices = builder.op.TopK("X", numpy.array([1], dtype=numpy.int64))
        self.assertIsInstance(values, str)
        self.assertEqual(builder.get_type(indices), onnx.TensorProto.INT64)

    def test_split_num_outputs_attribute_and_explicit_names(self):
        for names in (None, ["left", "right"]):
            with self.subTest(outputs=names):
                builder = self.make_builder(18)
                builder.make_tensor_input("X", 1, ("N", 4))
                outputs = builder.op.Split("X", axis=1, num_outputs=2, outputs=names)
                self.assertIsInstance(outputs, tuple)
                self.assertEqual(len(outputs), 2)
                if names is not None:
                    self.assertEqual(outputs, tuple(names))
                builder.make_tensor_output(outputs)
                model = builder.to_native(optimize=False)
                self.assertEqual(len(model.graph.node[0].output), 2)
                attributes = {str(a.name): a.i for a in model.graph.node[0].attribute}
                self.assertEqual(attributes["num_outputs"], 2)
                model = builder.to_onnx(optimize=False).proto
                checker.check_model(model)
                x = numpy.arange(8, dtype=numpy.float32).reshape(2, 4)
                left, right = ReferenceEvaluator(model).run(None, {"X": x})
                numpy.testing.assert_array_equal(left, x[:, :2])
                numpy.testing.assert_array_equal(right, x[:, 2:])
        builder = self.make_builder(18)
        builder.make_tensor_input("X", 1, ("N", 4))
        with self.assertRaisesRegex(ValueError, "must match"):
            builder.op.Split("X", outputs=["Y"], num_outputs=2)
        with self.assertRaisesRegex(ValueError, "requires outputs"):
            builder.op.Split("X")
        explicit_split = builder.op.Split(
            "X", numpy.array([1, 3], dtype=numpy.int64), axis=1, outputs=["Y", "Z"]
        )
        builder.make_tensor_output(explicit_split)
        checker.check_model(builder.to_onnx(optimize=False).proto)

    def test_native_attributes_and_graph_export(self):
        builder = self.make_builder(18)
        builder.make_tensor_input("X", 1, ("N", 3))
        output = builder.make_node(
            "Transpose", "X", attributes=[onnx.helper.make_attribute("perm", [1, 0])]
        )
        self.assertEqual(builder.get_shape(output), (3, "N"))
        builder.make_tensor_output(output)
        graph = builder.to_onnx(optimize=False, as_graph_proto=True).proto
        self.assertIsInstance(graph, onnx.GraphProto)
        self.assertEqual(graph.node[0].attribute[0].ints, [1, 0])

    def test_unknown_dimensions_and_explicit_rank(self):
        builder = self.make_builder(18)
        builder.make_tensor_input("X", 1, (None, 3))
        output = builder.op.Identity("X")
        self.assertEqual(builder.get_shape(output), (None, 3))
        builder.make_tensor_output(output)
        checker.check_model(builder.to_onnx(optimize=False).proto)
        builder = self.make_builder(18)
        builder.make_tensor_input("X", 1, None)
        self.assertFalse(builder.has_rank("X"))
        with self.assertRaisesRegex(NotImplementedError, "declared rank"):
            builder.op.Identity("X")
        builder.set_rank("X", 2)
        output = builder.op.Identity("X")
        self.assertEqual(builder.get_rank(output), 2)
        source = onnx.helper.make_model(
            onnx.helper.make_graph(
                [],
                "unranked",
                [onnx.helper.make_tensor_value_info("X", 1, None)],
                [onnx.helper.make_tensor_value_info("X", 1, None)],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )
        with self.assertRaisesRegex(NotImplementedError, "declared rank"):
            self.make_builder(source)

    def test_native_subgraph_attributes(self):
        builder = self.make_builder(18)
        builder.make_tensor_input("condition", onnx.TensorProto.BOOL, ())
        values = numpy.array([1, 2], dtype=numpy.float32)
        branch = onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "Constant", [], ["value"], value=onnx.numpy_helper.from_array(values)
                )
            ],
            "branch",
            [],
            [onnx.helper.make_tensor_value_info("value", 1, [2])],
        )
        output = builder.op.If("condition", then_branch=branch, else_branch=branch)
        self.assertEqual(builder.get_shape(output), (2,))
        builder.make_tensor_output(output)
        model = builder.to_onnx(optimize=False).proto
        checker.check_model(model)
        actual = ReferenceEvaluator(model).run(None, {"condition": numpy.array(True)})[0]
        numpy.testing.assert_array_equal(actual, values)

    def test_native_custom_shape_callback(self):
        builder = self.make_builder({"": 18, "custom": 1})
        builder.make_tensor_input("X", 1, ("N", 3))
        seen = []

        def callback(context, node):
            seen.append(str(node.op_type))
            context.set(str(node.output[0]), context.get(str(node.input[0])))

        builder.shapes_context.set_custom_shape_inference_function("custom", "Unknown", callback)
        output = builder.make_node("Unknown", ["X"], domain="custom")
        self.assertEqual(seen, ["Unknown"])
        self.assertEqual(builder.get_shape(output), ("N", 3))
        builder.make_tensor_output(output)
        checker.check_model(builder.to_onnx(optimize=False).proto)

    def test_sql_projection_and_filter(self):
        from yobx.builder.onnxlight import OnnxLightGraphBuilder
        from yobx.sql import sql_to_onnx

        artifact = sql_to_onnx(
            "SELECT a + b AS total FROM t WHERE a > 0",
            {"a": numpy.float32, "b": numpy.float32},
            target_opset=18,
            builder_cls=OnnxLightGraphBuilder,
        )
        checker.check_model(artifact.proto)
        a = numpy.array([1, -2, 3], dtype=numpy.float32)
        b = numpy.array([4, 5, 6], dtype=numpy.float32)
        actual = ReferenceEvaluator(artifact.proto).run(None, {"a": a, "b": b})[0]
        numpy.testing.assert_array_equal(actual, numpy.array([5, 9], dtype=numpy.float32))

    def test_numpy_tracing_proxy_inputs(self):
        from yobx.builder.onnxlight import OnnxLightGraphBuilder
        from yobx.sql import to_onnx

        def transform(x):
            return numpy.sqrt(numpy.abs(x) + numpy.float32(1))

        x = numpy.array([-1, 2, 3], dtype=numpy.float32)
        artifact = to_onnx(transform, x, builder_cls=OnnxLightGraphBuilder)
        checker.check_model(artifact.proto)
        actual = ReferenceEvaluator(artifact.proto).run(None, {"X": x})[0]
        numpy.testing.assert_allclose(actual, transform(x), rtol=1e-6)

    def test_unsupported_options_fail_explicitly(self):
        from yobx.builder.onnxlight import OnnxLightOptimizationOptions

        with self.assertRaisesRegex(TypeError, "OnnxLightOptimizationOptions"):
            self.make_builder(18, optimization_options=object())
        self.assertTrue(OnnxLightOptimizationOptions("default+onnxruntime").pattern_names())
        with self.assertRaisesRegex(ValueError, "Unsupported native patterns"):
            OnnxLightOptimizationOptions([object()])
        with self.assertRaises(ValueError):
            OnnxLightOptimizationOptions(max_iter=-2)
        builder = self.make_builder(18)
        with self.assertRaises(NotImplementedError):
            builder.to_onnx(large_model=True, as_graph_proto=True)
        with self.assertRaises(NotImplementedError):
            builder.to_onnx(mask_outputs=[True])
        with self.assertRaises(NotImplementedError):
            builder.set_sequence("S", 1)
        with self.assertRaisesRegex(ValueError, "Register the opset"):
            builder.make_node("Unknown", [], domain="unregistered")

    @unittest.skipUnless(
        "scikit-learn" in importlib.metadata.packages_distributions().get("sklearn", []),
        "scikit-learn is not installed",
    )
    def test_sklearn_linear_regression(self):
        from sklearn.linear_model import LinearRegression
        from yobx.builder.onnxlight import OnnxLightGraphBuilder
        from yobx.sklearn import to_onnx

        x = numpy.random.default_rng(0).normal(size=(20, 3)).astype(numpy.float32)
        y = x @ numpy.array([1, 2, 3], dtype=numpy.float32) + 4
        estimator = LinearRegression().fit(x, y)
        artifact = to_onnx(estimator, (x,), target_opset=18, builder_cls=OnnxLightGraphBuilder)
        checker.check_model(artifact.proto)
        for rows in (2, 7):
            inputs = x[:rows]
            actual = ReferenceEvaluator(artifact.proto).run(None, {"X": inputs})[0]
            numpy.testing.assert_allclose(
                actual.reshape(-1), estimator.predict(inputs), rtol=1e-5, atol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
