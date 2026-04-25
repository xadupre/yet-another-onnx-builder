import unittest
from yobx.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_transformers,
    skipif_ci_windows,
)


class TestValidateSummaryFields(ExtTestCase):
    """Tests that do not need torch/transformers — they only inspect dataclass fields."""

    def test_n_nodes_field_exists(self):
        """ValidateSummary exposes n_nodes and top_op_types fields."""
        from dataclasses import fields
        from yobx.torch.validate import ValidateSummary

        names = {f.name for f in fields(ValidateSummary)}
        self.assertIn("n_nodes", names)
        self.assertIn("top_op_types", names)

    def test_artifact_field_in_validate_data(self):
        """ValidateData exposes an artifact field."""
        from dataclasses import fields
        from yobx.torch.validate import ValidateData

        names = {f.name for f in fields(ValidateData)}
        self.assertIn("artifact", names)

    def test_summary_items_includes_n_nodes(self):
        """ValidateSummary.items() yields n_nodes when set."""
        from yobx.torch.validate import ValidateSummary

        s = ValidateSummary(model_id="m", prompt="p")
        s.n_nodes = 42
        d = dict(s.items())
        self.assertEqual(d["n_nodes"], 42)

    def test_summary_items_includes_top_op_types(self):
        """ValidateSummary.items() yields top_op_types when set."""
        from yobx.torch.validate import ValidateSummary

        s = ValidateSummary(model_id="m", prompt="p")
        s.top_op_types = "MatMul:5,Add:3"
        d = dict(s.items())
        self.assertEqual(d["top_op_types"], "MatMul:5,Add:3")

    def test_discrepancy_stats_fields_exist(self):
        """ValidateSummary exposes discrepancies_max_abs, atol, ratio_001, and ratio_01 fields."""
        from dataclasses import fields
        from yobx.torch.validate import ValidateSummary

        names = {f.name for f in fields(ValidateSummary)}
        self.assertIn("discrepancies_max_abs", names)
        self.assertIn("discrepancies_atol", names)
        self.assertIn("discrepancies_ratio_001", names)
        self.assertIn("discrepancies_ratio_01", names)

    def test_summary_items_includes_discrepancy_stats(self):
        """ValidateSummary.items() yields the new discrepancy stat fields when set."""
        from yobx.torch.validate import ValidateSummary

        s = ValidateSummary(model_id="m", prompt="p")
        s.discrepancies_max_abs = 0.005
        s.discrepancies_atol = 1e-4
        s.discrepancies_ratio_001 = 0.02
        s.discrepancies_ratio_01 = 0.0
        d = dict(s.items())
        self.assertAlmostEqual(d["discrepancies_max_abs"], 0.005)
        self.assertAlmostEqual(d["discrepancies_atol"], 1e-4)
        self.assertAlmostEqual(d["discrepancies_ratio_001"], 0.02)
        self.assertAlmostEqual(d["discrepancies_ratio_01"], 0.0)


@requires_torch("2.0")
@requires_transformers("5.0")
class TestValidateModel(ExtTestCase):
    def test_validate_model_tokenized_inputs_param(self):
        """validate_model accepts tokenized_inputs kwarg without error (no network needed)."""
        import inspect
        from yobx.torch.validate import validate_model

        sig = inspect.signature(validate_model)
        self.assertIn("tokenized_inputs", sig.parameters)
        p = sig.parameters["tokenized_inputs"]
        self.assertIsNone(p.default)

    def test_validate_model_config_overrides_param(self):
        """validate_model has config_overrides and random_weights parameters."""
        import inspect
        from yobx.torch.validate import validate_model

        sig = inspect.signature(validate_model)
        self.assertIn("config_overrides", sig.parameters)
        self.assertIsNone(sig.parameters["config_overrides"].default)
        self.assertIn("random_weights", sig.parameters)
        self.assertFalse(sig.parameters["random_weights"].default)

    def test_cmd_validate_has_random_weights(self):
        """CLI parser exposes --random-weights and --config-override flags."""
        from yobx._command_lines_parser import get_parser_validate

        parser = get_parser_validate()
        dest_names = {a.dest for a in parser._actions}
        self.assertIn("random_weights", dest_names)
        self.assertIn("config_override", dest_names)

    def test_default_prompt(self):
        from yobx.torch.validate import DEFAULT_PROMPT

        self.assertIsInstance(DEFAULT_PROMPT, str)
        self.assertGreater(len(DEFAULT_PROMPT), 0)

    def test_validate_model_import(self):
        from yobx.torch.validate import validate_model

        self.assertIsNotNone(validate_model)

    def test_cmd_validate_help(self):
        from yobx._command_lines_parser import get_parser_validate

        parser = get_parser_validate()
        self.assertIsNotNone(parser)

    def test_cmd_validate_registered(self):
        """Verify the 'validate' command is registered in the main parser."""
        from yobx._command_lines_parser import get_main_parser

        parser = get_main_parser()
        for action in parser._actions:
            if hasattr(action, "choices") and action.choices is not None:
                if "validate" in action.choices:
                    return
        self.fail("'validate' not found in main parser choices")

    def test_validate_model_tiny_llm(self):
        import torch
        from yobx.torch.validate import validate_model, ValidateSummary, ValidateData

        tokenized = {
            "input_ids": torch.randint(0, 1000, (1, 5), dtype=torch.int64),
            "attention_mask": torch.ones(1, 5, dtype=torch.int64),
        }
        summary, data = validate_model(
            "arnir0/Tiny-LLM",
            tokenized_inputs=tokenized,
            random_weights=True,
            max_new_tokens=3,
            do_run=False,
            verbose=0,
        )
        self.assertIsInstance(summary, ValidateSummary)
        self.assertIsInstance(data, ValidateData)
        self.assertEqual(summary.model_id, "arnir0/Tiny-LLM")
        self.assertEqual(summary.export, "OK")
        self.assertIsNotNone(data.observer)
        self.assertIsNotNone(data.kwargs)
        self.assertIsNotNone(data.dynamic_shapes)
        self.assertIsNotNone(data.filename)
        # Node stats should be populated after a successful export.
        self.assertIsNotNone(summary.n_nodes)
        self.assertGreater(summary.n_nodes, 0)
        self.assertIsNotNone(summary.top_op_types)
        self.assertGreater(len(summary.top_op_types), 0)

    def test_validate_model_captures_inputs(self):
        import torch
        from yobx.torch.validate import validate_model

        tokenized = {
            "input_ids": torch.randint(0, 1000, (1, 5), dtype=torch.int64),
            "attention_mask": torch.ones(1, 5, dtype=torch.int64),
        }
        _summary, data = validate_model(
            "arnir0/Tiny-LLM",
            tokenized_inputs=tokenized,
            random_weights=True,
            max_new_tokens=3,
            do_run=False,
            verbose=0,
        )
        observer = data.observer
        # The observer should have captured at least one input set
        self.assertGreater(len(observer.info), 0)

    def test_validate_model_exporter_dynamo(self):
        """validate_model with exporter='onnx-dynamo' runs without unhandled exception."""
        import torch
        from yobx.torch.validate import validate_model

        tokenized = {
            "input_ids": torch.randint(0, 1000, (1, 5), dtype=torch.int64),
            "attention_mask": torch.ones(1, 5, dtype=torch.int64),
        }
        summary, _data = validate_model(
            "arnir0/Tiny-LLM",
            exporter="onnx-dynamo",
            tokenized_inputs=tokenized,
            random_weights=True,
            max_new_tokens=3,
            do_run=False,
            quiet=True,
            verbose=0,
        )
        self.assertIsNotNone(summary.model_id)
        self.assertEqual(summary.model_id, "arnir0/Tiny-LLM")
        # Export may succeed or fail depending on torch version; both are acceptable.
        self.assertIsNotNone(summary.export)

    def test_validate_model_exporter_modelbuilder(self):
        """validate_model with exporter='modelbuilder' runs without unhandled exception."""
        import torch
        from yobx.torch.validate import validate_model

        tokenized = {
            "input_ids": torch.randint(0, 1000, (1, 5), dtype=torch.int64),
            "attention_mask": torch.ones(1, 5, dtype=torch.int64),
        }
        summary, _data = validate_model(
            "arnir0/Tiny-LLM",
            exporter="modelbuilder",
            tokenized_inputs=tokenized,
            random_weights=True,
            max_new_tokens=3,
            do_run=False,
            quiet=True,
            verbose=0,
        )
        self.assertIsNotNone(summary.model_id)
        self.assertEqual(summary.model_id, "arnir0/Tiny-LLM")
        # Export may succeed or fail depending on torch version; both are acceptable.
        self.assertIsNotNone(summary.export)

    def test_check_discrepancies_verbose_2(self):
        """_check_discrepancies prints per-row detail lines when verbose >= 2."""
        import io
        import sys
        import torch
        from yobx.torch.validate import validate_model

        tokenized = {
            "input_ids": torch.randint(0, 1000, (1, 5), dtype=torch.int64),
            "attention_mask": torch.ones(1, 5, dtype=torch.int64),
        }
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _summary, data = validate_model(
                "arnir0/Tiny-LLM",
                tokenized_inputs=tokenized,
                random_weights=True,
                max_new_tokens=3,
                do_run=True,
                quiet=True,
                verbose=2,
            )
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        # Verbose >= 1 should always print the summary line.
        self.assertIn("[validate_model] discrepancies:", output)
        # Verbose >= 2 should print per-row status lines.
        if data.discrepancies:
            self.assertIn("[0]", output)

    def test_check_discrepancies_verbose_3(self):
        """_check_discrepancies prints tensor shapes when verbose >= 3."""
        import io
        import sys
        import torch
        from yobx.torch.validate import validate_model

        tokenized = {
            "input_ids": torch.randint(0, 1000, (1, 5), dtype=torch.int64),
            "attention_mask": torch.ones(1, 5, dtype=torch.int64),
        }
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _summary, data = validate_model(
                "arnir0/Tiny-LLM",
                tokenized_inputs=tokenized,
                random_weights=True,
                max_new_tokens=3,
                do_run=True,
                quiet=True,
                verbose=3,
            )
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        self.assertIn("[validate_model] discrepancies:", output)
        # Verbose >= 3: tensor shape strings should appear when discrepancies exist.
        if data.discrepancies and data.discrepancies[0].get("inputs"):
            self.assertIn("inputs:", output)

    def test_validate_model_node_stats_in_summary(self):
        """validate_model populates n_nodes and top_op_types in summary after export."""
        import torch
        from yobx.torch.validate import validate_model, ValidateSummary

        tokenized = {
            "input_ids": torch.randint(0, 1000, (1, 5), dtype=torch.int64),
            "attention_mask": torch.ones(1, 5, dtype=torch.int64),
        }
        summary, _data = validate_model(
            "arnir0/Tiny-LLM",
            tokenized_inputs=tokenized,
            random_weights=True,
            max_new_tokens=3,
            do_run=False,
            quiet=True,
            verbose=0,
        )
        self.assertIsInstance(summary, ValidateSummary)
        if summary.export == "OK":
            self.assertIsNotNone(summary.n_nodes)
            self.assertIsNotNone(summary.top_op_types)
            # top_op_types should look like "OpType:N,..." with a colon separator.
            self.assertIn(":", summary.top_op_types)

    def test_validate_model_discrepancies_in_artifact_report(self):
        """After do_run=True, the artifact report includes the discrepancy rows."""
        import torch
        from yobx.container import ExportArtifact
        from yobx.torch.validate import validate_model

        tokenized = {
            "input_ids": torch.randint(0, 1000, (1, 5), dtype=torch.int64),
            "attention_mask": torch.ones(1, 5, dtype=torch.int64),
        }
        _summary, data = validate_model(
            "arnir0/Tiny-LLM",
            tokenized_inputs=tokenized,
            random_weights=True,
            max_new_tokens=3,
            do_run=True,
            quiet=True,
            verbose=0,
        )
        if not isinstance(data.artifact, ExportArtifact):
            return  # dynamo exporter — skip
        self.assertIsNotNone(data.artifact.report)
        # Discrepancies from check_discrepancies should be stored in the report.
        if data.discrepancies:
            self.assertEqual(len(data.artifact.report.discrepancies), len(data.discrepancies))

    @skipif_ci_windows("xlsx file locked by another process on Windows")
    def test_validate_model_artifact_xlsx_has_discrepancies_sheet(self):
        """The xlsx saved alongside the ONNX contains a 'discrepancies' sheet."""
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            return
        import os
        import tempfile
        import torch
        import pandas
        from yobx.container import ExportArtifact
        from yobx.torch.validate import validate_model

        tokenized = {
            "input_ids": torch.randint(0, 1000, (1, 5), dtype=torch.int64),
            "attention_mask": torch.ones(1, 5, dtype=torch.int64),
        }
        with tempfile.TemporaryDirectory() as tmp:
            _summary, data = validate_model(
                "arnir0/Tiny-LLM",
                tokenized_inputs=tokenized,
                random_weights=True,
                max_new_tokens=3,
                do_run=True,
                quiet=True,
                verbose=0,
                dump_folder=tmp,
            )
            if not isinstance(data.artifact, ExportArtifact):
                return  # dynamo exporter — skip
            if data.filename is None:
                return
            xlsx_path = os.path.splitext(data.filename)[0] + ".xlsx"
            if not os.path.exists(xlsx_path):
                return
            sheets = pandas.ExcelFile(xlsx_path).sheet_names
            if data.discrepancies:
                self.assertIn("discrepancies", sheets)


if __name__ == "__main__":
    unittest.main(verbosity=2)
