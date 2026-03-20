"""
Unit tests for :class:`~yobx.container.ExportArtifact` and
:class:`~yobx.container.ExportReport`.
"""

import os
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
from yobx.container import ExportArtifact, ExportReport, FunctionPieces
from yobx.ext_test_case import ExtTestCase


def _make_simple_model():
    """Return a minimal ModelProto (identity on a float32 vector)."""
    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["N", 4])
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["N", 4])
    node = oh.make_node("Identity", ["X"], ["Y"])
    graph = oh.make_graph([node], "g", [X], [Y])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    return model


class TestExportReport(ExtTestCase):
    def test_empty(self):
        r = ExportReport()
        self.assertEqual(r.stats, [])
        self.assertEqual(r.extra, {})

    def test_stats_stored(self):
        stats = [{"pattern": "p1", "added": 1, "removed": 0, "time_in": 0.01}]
        r = ExportReport(stats=stats)
        self.assertEqual(len(r.stats), 1)
        self.assertEqual(r.stats[0]["pattern"], "p1")

    def test_update_extra(self):
        r = ExportReport()
        r.update({"time_total": 0.42, "source": "test"})
        self.assertIn("time_total", r.extra)
        self.assertAlmostEqual(r.extra["time_total"], 0.42)

    def test_to_dict(self):
        r = ExportReport(stats=[{"pattern": "a"}], extra={"k": "v"})
        d = r.to_dict()
        self.assertIn("stats", d)
        self.assertIn("extra", d)
        self.assertEqual(d["extra"]["k"], "v")

    def test_repr(self):
        r = ExportReport(stats=[{"x": 1}, {"x": 2}], extra={"k": "v"})
        text = repr(r)
        self.assertIn("ExportReport", text)
        self.assertIn("n_stats=2", text)


class TestExportArtifact(ExtTestCase):
    def _artifact(self):
        model = _make_simple_model()
        return ExportArtifact(proto=model, report=ExportReport())

    def test_attributes(self):
        artifact = self._artifact()
        self.assertIsNotNone(artifact.proto)
        self.assertIsNone(artifact.container)
        self.assertIsNone(artifact.filename)
        self.assertIsInstance(artifact.report, ExportReport)

    def test_filename_stored(self):
        artifact = ExportArtifact(proto=_make_simple_model(), filename="model.onnx")
        self.assertEqual(artifact.filename, "model.onnx")

    def test_get_proto_no_container(self):
        artifact = self._artifact()
        proto = artifact.get_proto(include_weights=True)
        self.assertIsInstance(proto, onnx.ModelProto)
        # Same object when there is no container
        self.assertIs(proto, artifact.proto)

    def test_get_proto_no_container_no_weights(self):
        artifact = self._artifact()
        proto = artifact.get_proto(include_weights=False)
        self.assertIsInstance(proto, onnx.ModelProto)
        self.assertIs(proto, artifact.proto)

    def test_save(self):
        artifact = self._artifact()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            returned = artifact.save(path)
            self.assertTrue(os.path.exists(path))
            self.assertIsInstance(returned, onnx.ModelProto)
            self.assertEqual(artifact.filename, path)

    def test_save_non_model_proto_raises(self):
        """Saving a FunctionProto directly should raise TypeError."""
        import os
        import tempfile

        artifact = ExportArtifact(proto=onnx.FunctionProto())

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fn.onnx")
            with self.assertRaises(TypeError):
                artifact.save(path)

    def test_repr(self):
        artifact = self._artifact()
        text = repr(artifact)
        self.assertIn("ExportArtifact", text)
        self.assertIn("ModelProto", text)
        self.assertIn("filename=None", text)
        self.assertIn("container=None", text)

    def test_load(self):
        artifact = self._artifact()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            artifact.save(path)
            loaded = ExportArtifact.load(path)
            self.assertIsInstance(loaded, ExportArtifact)
            self.assertIsInstance(loaded.proto, onnx.ModelProto)
            self.assertIsNone(loaded.container)
            self.assertEqual(loaded.filename, path)

    def test_reference_evaluator_accepts_artifact(self):
        """ExtendedReferenceEvaluator should accept an ExportArtifact."""
        from yobx.reference import ExtendedReferenceEvaluator

        artifact = self._artifact()
        ref = ExtendedReferenceEvaluator(artifact)
        X = np.random.randn(3, 4).astype(np.float32)
        (Y,) = ref.run(None, {"X": X})
        np.testing.assert_allclose(Y, X)

    def test_from_sql_to_onnx(self):
        """sql_to_onnx should return an ExportArtifact."""
        from yobx.sql import sql_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        dtypes = {"a": np.float32, "b": np.float32}
        artifact = sql_to_onnx("SELECT a + b AS total FROM t", dtypes)
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsInstance(artifact.report, ExportReport)
        self.assertIsInstance(artifact.proto, onnx.ModelProto)
        self.assertIsNone(artifact.container)

        # Can be passed to ExtendedReferenceEvaluator
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b, rtol=1e-5)

    def test_to_onnx_function_returns_artifact(self):
        """to_onnx with export_as_function=True should return an ExportArtifact with function field."""
        from yobx.xbuilder import GraphBuilder, FunctionOptions
        from yobx.reference import ExtendedReferenceEvaluator

        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        g.op.Add("X", "X", outputs=["Y"])
        g.make_tensor_output("Y", is_dimension=False, indexed=False)

        artifact = g.to_onnx(
            function_options=FunctionOptions(
                export_as_function=True, name="double", domain="test_domain"
            ),
            inline=False,
        )

        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsInstance(artifact.proto, onnx.FunctionProto)
        self.assertIsNone(artifact.container)
        # function field is always set for function exports
        self.assertIsNotNone(artifact.function)
        self.assertIsInstance(artifact.function, FunctionPieces)
        # No initializers in this simple function
        self.assertIsNone(artifact.function.initializers_name)
        self.assertIsNone(artifact.function.initializers_dict)
        self.assertIsNone(artifact.function.initializers_renaming)

        # Proto should be usable with ExtendedReferenceEvaluator
        ref = ExtendedReferenceEvaluator(artifact)
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        (Y,) = ref.run(None, {"X": X})
        np.testing.assert_allclose(Y, X + X)

    def test_to_onnx_function_with_initializers_returns_artifact(self):
        """to_onnx with export_as_function=True and return_initializer=True."""
        from yobx.xbuilder import GraphBuilder, FunctionOptions

        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        np_weights = np.arange(6).reshape((2, 3)).astype(np.float32)
        init = g.make_initializer("weights", np_weights)
        g.op.MatMul("X", init, outputs=["Y"])
        g.make_tensor_output("Y", is_dimension=False, indexed=False)

        artifact = g.to_onnx(
            function_options=FunctionOptions(
                export_as_function=True,
                name="linear",
                domain="test_domain",
                return_initializer=True,
            ),
            inline=False,
        )

        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsInstance(artifact.proto, onnx.FunctionProto)
        # function field carries the initializer data
        self.assertIsNotNone(artifact.function)
        self.assertIsInstance(artifact.function, FunctionPieces)
        self.assertIsNotNone(artifact.function.initializers_name)
        self.assertIsNotNone(artifact.function.initializers_dict)
        self.assertIsNotNone(artifact.function.initializers_renaming)
        self.assertGreater(len(artifact.function.initializers_name), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
