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
from yobx.ext_test_case import ExtTestCase, skipif_ci_windows


def _make_simple_model():
    """Return a minimal ModelProto (identity on a float32 vector)."""
    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["N", 4])
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["N", 4])
    node = oh.make_node("Identity", ["X"], ["Y"])
    graph = oh.make_graph([node], "g", [X], [Y])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    return model


def _make_relu_model():
    """Return a ModelProto with a Relu node (static shapes)."""
    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 8])
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 8])
    node = oh.make_node("Relu", ["X"], ["Y"])
    graph = oh.make_graph([node], "g", [X], [Y])
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])


def _make_matmul_model():
    """Return a ModelProto with a MatMul + Add node."""
    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 4])
    W = oh.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [4, 2])
    B = oh.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [3, 2])
    Z = oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [3, 2])
    mm = oh.make_node("MatMul", ["X", "W"], ["T"])
    add = oh.make_node("Add", ["T", "B"], ["Z"])
    graph = oh.make_graph([mm, add], "g", [X, W, B], [Z])
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])


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

    def test_to_string_empty(self):
        r = ExportReport()
        text = r.to_string()
        self.assertIsInstance(text, str)

    def test_to_string_with_extra(self):
        r = ExportReport(extra={"time_total": 0.42, "source": "test"})
        text = r.to_string()
        self.assertIn("extra", text)
        self.assertIn("time_total", text)
        self.assertIn("0.42", text)

    def test_to_string_with_stats(self):
        stats = [
            {"pattern": "p1", "added": 1, "removed": 2, "time_in": 0.01},
            {"pattern": "p1", "added": 0, "removed": 1, "time_in": 0.02},
            {"pattern": "p2", "added": 2, "removed": 0, "time_in": 0.005},
        ]
        r = ExportReport(stats=stats)
        text = r.to_string()
        self.assertIn("stats", text)
        self.assertIn("p1", text)

    def test_to_string_with_build_stats(self):
        from yobx.container import BuildStats

        bs = BuildStats()
        bs["time_export_write_model"] = 0.5
        r = ExportReport(build_stats=bs)
        text = r.to_string()
        self.assertIn("build_stats", text)
        self.assertIn("time_export_write_model", text)

    @skipif_ci_windows("issue with excel")
    def test_to_excel(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            return
        import pandas

        stats = [
            {"pattern": "p1", "added": 1, "removed": 2, "time_in": 0.01},
            {"pattern": "p2", "added": 0, "removed": 1, "time_in": 0.02},
        ]
        r = ExportReport(stats=stats, extra={"time_total": 0.42})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.xlsx")
            r.to_excel(path)
            self.assertTrue(os.path.exists(path))
            # Verify sheets
            sheets = pandas.ExcelFile(path).sheet_names
            self.assertIn("stats", sheets)
            self.assertIn("stats_agg", sheets)
            self.assertIn("extra", sheets)
            # Read back stats
            df = pandas.read_excel(path, sheet_name="stats")
            self.assertEqual(len(df), 2)
            # Read back extra
            df_extra = pandas.read_excel(path, sheet_name="extra")
            self.assertIn("key", df_extra.columns)
            self.assertIn("value", df_extra.columns)

    @skipif_ci_windows("issue with excel")
    def test_to_excel_with_build_stats(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            return
        import pandas
        from yobx.container import BuildStats

        bs = BuildStats()
        bs["time_export_write_model"] = 0.5
        r = ExportReport(extra={"k": "v"}, build_stats=bs)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report_bs.xlsx")
            r.to_excel(path)
            self.assertTrue(os.path.exists(path))
            sheets = pandas.ExcelFile(path).sheet_names
            self.assertIn("build_stats", sheets)
            df_bs = pandas.read_excel(path, sheet_name="build_stats")
            self.assertEqual(list(df_bs.columns), ["key", "value"])

    # ------------------------------------------------------------------
    # node_stats tests
    # ------------------------------------------------------------------

    def test_node_stats_empty_by_default(self):
        r = ExportReport()
        self.assertEqual(r.node_stats, [])

    def test_node_stats_init(self):
        rows = [{"op_type": "Relu", "count": 2, "flops": 16}]
        r = ExportReport(node_stats=rows)
        self.assertEqual(len(r.node_stats), 1)
        self.assertEqual(r.node_stats[0]["op_type"], "Relu")

    def test_compute_node_stats_identity(self):
        """Identity node should appear in node_stats with count=1."""
        r = ExportReport()
        r.compute_node_stats(_make_simple_model())
        self.assertGreater(len(r.node_stats), 0)
        op_types = {row["op_type"] for row in r.node_stats}
        self.assertIn("Identity", op_types)
        identity_row = next(row for row in r.node_stats if row["op_type"] == "Identity")
        self.assertEqual(identity_row["count"], 1)

    def test_compute_node_stats_relu_static(self):
        """Relu on a fully static shape should give a non-None flops estimate."""
        r = ExportReport()
        r.compute_node_stats(_make_relu_model())
        relu_rows = [row for row in r.node_stats if row["op_type"] == "Relu"]
        self.assertEqual(len(relu_rows), 1)
        self.assertEqual(relu_rows[0]["count"], 1)
        # Relu is elementwise: 4*8 = 32 elements
        self.assertEqual(relu_rows[0]["flops"], 32)

    def test_compute_node_stats_matmul(self):
        """MatMul and Add should both appear with correct counts and FLOPs."""
        r = ExportReport()
        r.compute_node_stats(_make_matmul_model())
        op_types = {row["op_type"] for row in r.node_stats}
        self.assertIn("MatMul", op_types)
        self.assertIn("Add", op_types)
        mm_row = next(row for row in r.node_stats if row["op_type"] == "MatMul")
        add_row = next(row for row in r.node_stats if row["op_type"] == "Add")
        self.assertEqual(mm_row["count"], 1)
        self.assertEqual(add_row["count"], 1)
        # FLOPs should be positive integers for static shapes
        self.assertIsNotNone(mm_row["flops"])
        self.assertIsNotNone(add_row["flops"])

    def test_compute_node_stats_chained(self):
        """compute_node_stats returns self (method chaining)."""
        r = ExportReport()
        result = r.compute_node_stats(_make_relu_model())
        self.assertIs(result, r)

    def test_repr_includes_n_node_stats(self):
        rows = [{"op_type": "Relu", "count": 1, "flops": 32}]
        r = ExportReport(node_stats=rows)
        text = repr(r)
        self.assertIn("n_node_stats=1", text)

    def test_to_dict_includes_node_stats(self):
        rows = [{"op_type": "Add", "count": 3, "flops": 96}]
        r = ExportReport(node_stats=rows)
        d = r.to_dict()
        self.assertIn("node_stats", d)
        self.assertEqual(len(d["node_stats"]), 1)

    def test_to_string_includes_node_stats(self):
        r = ExportReport()
        r.compute_node_stats(_make_relu_model())
        text = r.to_string()
        self.assertIn("node_stats", text)
        self.assertIn("Relu", text)

    def test_update_merges_node_stats(self):
        r1 = ExportReport(node_stats=[{"op_type": "Relu", "count": 1, "flops": 32}])
        r2 = ExportReport(node_stats=[{"op_type": "Add", "count": 2, "flops": 64}])
        r1.update(r2)
        self.assertEqual(len(r1.node_stats), 2)

    @skipif_ci_windows("issue with excel")
    def test_to_excel_with_node_stats(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            return
        import pandas

        r = ExportReport()
        r.compute_node_stats(_make_matmul_model())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report_ns.xlsx")
            r.to_excel(path)
            self.assertTrue(os.path.exists(path))
            sheets = pandas.ExcelFile(path).sheet_names
            self.assertIn("node_stats", sheets)
            df_ns = pandas.read_excel(path, sheet_name="node_stats")
            self.assertIn("op_type", df_ns.columns)
            self.assertIn("count", df_ns.columns)
            self.assertIn("flops", df_ns.columns)
            op_types = set(df_ns["op_type"].tolist())
            self.assertIn("MatMul", op_types)
            self.assertIn("Add", op_types)

    # ------------------------------------------------------------------
    # symbolic_flops tests
    # ------------------------------------------------------------------

    def test_symbolic_flops_empty_by_default(self):
        r = ExportReport()
        self.assertEqual(r.symbolic_flops, [])

    def test_symbolic_flops_init(self):
        rows = [{"op_type": "Relu", "node_name": "relu_0", "symbolic_flops": "batch*d"}]
        r = ExportReport(symbolic_flops=rows)
        self.assertEqual(len(r.symbolic_flops), 1)
        self.assertEqual(r.symbolic_flops[0]["op_type"], "Relu")

    def test_compute_symbolic_flops_static(self):
        """Relu on a fully static shape should give an integer flops estimate."""
        r = ExportReport()
        r.compute_symbolic_flops(_make_relu_model())
        self.assertEqual(len(r.symbolic_flops), 1)
        row = r.symbolic_flops[0]
        self.assertEqual(row["op_type"], "Relu")
        # Relu is elementwise: 4*8 = 32 elements
        self.assertEqual(row["symbolic_flops"], 32)

    def test_compute_symbolic_flops_symbolic(self):
        """Relu on a symbolic shape should give a non-None symbolic expression."""
        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["batch", "d"])
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["batch", "d"])
        node = oh.make_node("Relu", ["X"], ["Y"])
        graph = oh.make_graph([node], "g", [X], [Y])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])

        r = ExportReport()
        r.compute_symbolic_flops(model)
        self.assertEqual(len(r.symbolic_flops), 1)
        row = r.symbolic_flops[0]
        self.assertEqual(row["op_type"], "Relu")
        # Should be a non-trivial string expression involving the symbolic dims
        self.assertIsNotNone(row["symbolic_flops"])
        self.assertIsInstance(row["symbolic_flops"], str)

    def test_compute_symbolic_flops_matmul(self):
        """MatMul and Add nodes should each appear with flops entries."""
        r = ExportReport()
        r.compute_symbolic_flops(_make_matmul_model())
        self.assertEqual(len(r.symbolic_flops), 2)
        op_types = [row["op_type"] for row in r.symbolic_flops]
        self.assertIn("MatMul", op_types)
        self.assertIn("Add", op_types)
        for row in r.symbolic_flops:
            self.assertIn("node_name", row)
            self.assertIn("symbolic_flops", row)

    def test_compute_symbolic_flops_chained(self):
        """compute_symbolic_flops returns self (method chaining)."""
        r = ExportReport()
        result = r.compute_symbolic_flops(_make_relu_model())
        self.assertIs(result, r)

    def test_repr_includes_n_symbolic_flops(self):
        rows = [{"op_type": "Relu", "node_name": "", "symbolic_flops": 32}]
        r = ExportReport(symbolic_flops=rows)
        text = repr(r)
        self.assertIn("n_symbolic_flops=1", text)

    def test_to_dict_includes_symbolic_flops(self):
        rows = [{"op_type": "Add", "node_name": "", "symbolic_flops": "a*b"}]
        r = ExportReport(symbolic_flops=rows)
        d = r.to_dict()
        self.assertIn("symbolic_flops", d)
        self.assertEqual(len(d["symbolic_flops"]), 1)

    def test_to_string_includes_symbolic_flops(self):
        r = ExportReport()
        r.compute_symbolic_flops(_make_relu_model())
        text = r.to_string()
        self.assertIn("symbolic_flops", text)
        self.assertIn("Relu", text)

    def test_update_merges_symbolic_flops(self):
        r1 = ExportReport(
            symbolic_flops=[{"op_type": "Relu", "node_name": "", "symbolic_flops": 32}]
        )
        r2 = ExportReport(
            symbolic_flops=[{"op_type": "Add", "node_name": "", "symbolic_flops": 64}]
        )
        r1.update(r2)
        self.assertEqual(len(r1.symbolic_flops), 2)

    @skipif_ci_windows("issue with excel")
    def test_to_excel_with_symbolic_flops(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            return
        import pandas

        r = ExportReport()
        r.compute_symbolic_flops(_make_matmul_model())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report_sf.xlsx")
            r.to_excel(path)
            self.assertTrue(os.path.exists(path))
            sheets = pandas.ExcelFile(path).sheet_names
            self.assertIn("symbolic_flops", sheets)
            df_sf = pandas.read_excel(path, sheet_name="symbolic_flops")
            self.assertIn("op_type", df_sf.columns)
            self.assertIn("node_name", df_sf.columns)
            self.assertIn("symbolic_flops", df_sf.columns)
            op_types = set(df_sf["op_type"].tolist())
            self.assertIn("MatMul", op_types)
            self.assertIn("Add", op_types)


class TestExportArtifact(ExtTestCase):
    def _artifact(self):
        model = _make_simple_model()
        report = ExportReport(
            stats=[{"pattern": "p1", "added": 1, "removed": 2, "time_in": 0.01}],
            extra={"time_total": 0.1},
        )
        return ExportArtifact(proto=model, report=report)

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
            # A report Excel file should be produced alongside the .onnx file.
            excel_path = os.path.join(tmp, "model.xlsx")
            self.assertTrue(os.path.exists(excel_path))

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
        artifact = sql_to_onnx(
            "SELECT a + b AS total FROM t", dtypes, return_optimize_report=True
        )
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
        from yobx.xbuilder import GraphBuilder, FunctionOptions
        from yobx.reference import ExtendedReferenceEvaluator

        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        g.op.Add("X", "X", outputs=["Y"])
        g.make_tensor_output("Y", indexed=False)

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
        g.make_tensor_output("Y", indexed=False)

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

    def test_compute_node_stats_on_artifact(self):
        """ExportArtifact.compute_node_stats populates report.node_stats."""
        artifact = ExportArtifact(proto=_make_matmul_model())
        result = artifact.compute_node_stats()
        self.assertIs(result, artifact)
        self.assertIsNotNone(artifact.report)
        self.assertIsInstance(artifact.report, ExportReport)
        self.assertGreater(len(artifact.report.node_stats), 0)
        op_types = {row["op_type"] for row in artifact.report.node_stats}
        self.assertIn("MatMul", op_types)
        self.assertIn("Add", op_types)

    def test_compute_node_stats_no_op_on_function_proto(self):
        """compute_node_stats on a FunctionProto artifact does nothing."""
        artifact = ExportArtifact(proto=onnx.FunctionProto())
        artifact.compute_node_stats()
        # report stays None (or empty node_stats) — no crash
        if artifact.report is not None:
            self.assertEqual(artifact.report.node_stats, [])

    def test_compute_symbolic_flops_on_artifact(self):
        """ExportArtifact.compute_symbolic_flops populates report.symbolic_flops."""
        artifact = ExportArtifact(proto=_make_matmul_model())
        result = artifact.compute_symbolic_flops()
        self.assertIs(result, artifact)
        self.assertIsNotNone(artifact.report)
        self.assertIsInstance(artifact.report, ExportReport)
        self.assertGreater(len(artifact.report.symbolic_flops), 0)
        op_types = [row["op_type"] for row in artifact.report.symbolic_flops]
        self.assertIn("MatMul", op_types)
        self.assertIn("Add", op_types)

    def test_compute_symbolic_flops_no_op_on_function_proto(self):
        """compute_symbolic_flops on a FunctionProto artifact does nothing."""
        artifact = ExportArtifact(proto=onnx.FunctionProto())
        artifact.compute_symbolic_flops()
        if artifact.report is not None:
            self.assertEqual(artifact.report.symbolic_flops, [])

    @skipif_ci_windows("issue with excel")
    def test_save_includes_node_stats_in_excel(self):
        """save() should produce an Excel file with both node_stats
        and symbolic_flops sheets populated automatically."""
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            return
        import pandas

        artifact = ExportArtifact(
            proto=_make_matmul_model(), report=ExportReport(extra={"time_total": 0.1})
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            artifact.save(path)
            excel_path = os.path.join(tmp, "model.xlsx")
            self.assertTrue(os.path.exists(excel_path))
            sheets = pandas.ExcelFile(excel_path).sheet_names
            self.assertIn("node_stats", sheets)
            df_ns = pandas.read_excel(excel_path, sheet_name="node_stats")
            self.assertIn("op_type", df_ns.columns)
            self.assertIn("count", df_ns.columns)
            op_types = set(df_ns["op_type"].tolist())
            self.assertIn("MatMul", op_types)
            self.assertIn("Add", op_types)
            self.assertIn("symbolic_flops", sheets)
            df_sf = pandas.read_excel(excel_path, sheet_name="symbolic_flops")
            self.assertIn("op_type", df_sf.columns)
            self.assertIn("symbolic_flops", df_sf.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
