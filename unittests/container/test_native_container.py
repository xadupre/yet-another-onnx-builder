"""Validates native artifacts and external-data ownership without reference ONNX."""

import os
import tempfile
import unittest

import numpy
from onnx_light import onnx
from onnx_light.onnx import helper, numpy_helper
from onnx_light.onnx.reference import ReferenceEvaluator

from yobx.container import ExportArtifact, ExtendedModelContainer, make_large_tensor_proto
from yobx.container.model_container import _get_all_tensors, uses_external_data


class TestNativeContainer(unittest.TestCase):
    def make_container(self):
        first = numpy.arange(4, dtype=numpy.float32).reshape(2, 2)
        second = numpy.full((2, 2), 3, dtype=numpy.float32)
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Add", ["X", "first"], ["T"]),
                    helper.make_node("Add", ["T", "second"], ["Y"]),
                ],
                "external",
                [helper.make_tensor_value_info("X", 1, [2, 2])],
                [helper.make_tensor_value_info("Y", 1, [2, 2])],
                [
                    make_large_tensor_proto("#first", "first", 1, [2, 2]),
                    make_large_tensor_proto("#second", "second", 1, [2, 2]),
                    numpy_helper.from_array(numpy.array(["native"], dtype=object), "labels"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=8,
        )
        model.metadata_props.add(key="owner", value="native")
        return ExtendedModelContainer(model, {"#first": first, "#second": second}), first + second

    def test_external_data_roundtrips(self):
        for together in (True, False):
            with self.subTest(all_tensors_to_one_file=together):
                container, expected = self.make_container()
                artifact = ExportArtifact(proto=container.model_proto, container=container)
                original = artifact.SerializeToString()
                with tempfile.TemporaryDirectory(dir=".") as directory:
                    path = os.path.join(directory, "model.onnx")
                    saved = artifact.save(path, all_tensors_to_one_file=together)
                    self.assertIsInstance(saved, onnx.ModelProto)
                    self.assertEqual(artifact.SerializeToString(), original)
                    loaded = ExportArtifact.load(path)
                    self.assertIsInstance(loaded.proto, onnx.ModelProto)
                    self.assertIsInstance(loaded.container, ExtendedModelContainer)
                    self.assertEqual(len(loaded.container.large_initializers), 2)
                    self.assertEqual(
                        [(entry.key, entry.value) for entry in loaded.metadata_props],
                        [("owner", "native")],
                    )
                    embedded = loaded.get_proto()
                    self.assertFalse(
                        any(uses_external_data(t) for t in _get_all_tensors(embedded))
                    )
                    actual = ReferenceEvaluator(embedded).run(
                        None, {"X": numpy.zeros((2, 2), dtype=numpy.float32)}
                    )[0]
                    numpy.testing.assert_array_equal(actual, expected)
                    strings = next(t for t in embedded.graph.initializer if t.name == "labels")
                    numpy.testing.assert_array_equal(numpy_helper.to_array(strings), ["native"])
                    deferred = ExportArtifact.load(path, load_large_initializers=False)
                    self.assertEqual(deferred.container.large_initializers, {})
                    with self.assertRaisesRegex(RuntimeError, "Unable to find"):
                        deferred.get_proto()
                    deferred.container._load_large_initializers(path)
                    self.assertEqual(len(deferred.container.large_initializers), 2)
                    self.assertIsInstance(deferred.get_proto(), onnx.ModelProto)
                    reserialized = os.path.join(directory, "resaved.onnx")
                    loaded.save(reserialized, all_tensors_to_one_file=not together)
                    self.assertIsInstance(
                        ExportArtifact.load(reserialized).get_proto(), onnx.ModelProto
                    )

    def test_nested_attribute_external_data(self):
        def branch(name):
            return helper.make_graph(
                [
                    helper.make_node(
                        "Constant",
                        [],
                        ["value"],
                        value=make_large_tensor_proto(f"#{name}", "weight", 1, [2]),
                    )
                ],
                name,
                [],
                [helper.make_tensor_value_info("value", 1, [2])],
            )

        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "If",
                        ["condition"],
                        ["Y"],
                        then_branch=branch("then"),
                        else_branch=branch("else"),
                    )
                ],
                "nested",
                [helper.make_tensor_value_info("condition", 9, [])],
                [helper.make_tensor_value_info("Y", 1, [2])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=8,
        )
        values = {
            "#then": numpy.array([1, 2], dtype=numpy.float32),
            "#else": numpy.array([3, 4], dtype=numpy.float32),
        }
        container = ExtendedModelContainer(model, values)
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = os.path.join(directory, "nested.onnx")
            container.save(path)
            restored = ExtendedModelContainer().load(path)
            self.assertEqual(len(restored.large_initializers), 2)
            embedded = restored.get_model_with_data()
            self.assertFalse(any(uses_external_data(t) for t in _get_all_tensors(embedded)))
            actual = ReferenceEvaluator(embedded).run(None, {"condition": numpy.array(True)})[0]
            numpy.testing.assert_array_equal(actual, values["#then"])

    def test_native_builder_large_export(self):
        from yobx.builder.onnxlight import OnnxLightGraphBuilder

        builder = OnnxLightGraphBuilder(18)
        builder.make_tensor_input("X", 1, ("batch", 3))
        weights = numpy.array([1, 2, 3], dtype=numpy.float32)
        output = builder.op.Add("X", weights)
        builder.make_tensor_output(output)
        artifact = builder.to_onnx(large_model=True, external_threshold=0, optimize=False)
        self.assertIsInstance(artifact.container, ExtendedModelContainer)
        self.assertTrue(uses_external_data(artifact.proto.graph.initializer[0]))
        self.assertEqual(len(artifact.container.large_initializers), 1)
        self.assertFalse(uses_external_data(artifact.get_proto().graph.initializer[0]))
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = os.path.join(directory, "large.onnx")
            artifact.save(path)
            loaded = ExportArtifact.load(path)
            actual = ReferenceEvaluator(loaded.get_proto()).run(
                None, {"X": numpy.zeros((2, 3), dtype=numpy.float32)}
            )[0]
            numpy.testing.assert_array_equal(actual, numpy.broadcast_to(weights, (2, 3)))

    def test_invalid_external_storage(self):
        container, _ = self.make_container()
        with self.assertRaises(TypeError):
            container.model_proto = object()
        with self.assertRaises(TypeError):
            ExportArtifact(proto=object())
        with self.assertRaises(TypeError):
            ExportArtifact(container=object())
        with self.assertRaises(NotImplementedError):
            make_large_tensor_proto("#strings", "strings", onnx.TensorProto.STRING, [1])
        with self.assertRaises(ValueError):
            _ = ExtendedModelContainer().model_proto
        with self.assertRaises(ValueError):
            container.externalize_initializers(-1)
        with self.assertRaises(ValueError):
            container.externalize_initializers(0)
        with self.assertRaises(TypeError):
            container.get_raw_data(object())
        with self.assertRaises(NotImplementedError):
            container.to_ir()


if __name__ == "__main__":
    unittest.main()
