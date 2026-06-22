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
        """ValidateSummary exposes discrepancies_max_abs, atol, rtol, ratio_001, and ratio_01 fields."""  # noqa: E501
        from dataclasses import fields
        from yobx.torch.validate import ValidateSummary

        names = {f.name for f in fields(ValidateSummary)}
        self.assertIn("discrepancies_max_abs", names)
        self.assertIn("discrepancies_atol", names)
        self.assertIn("discrepancies_rtol", names)
        self.assertIn("discrepancies_ratio_001", names)
        self.assertIn("discrepancies_ratio_01", names)

    def test_summary_items_includes_discrepancy_stats(self):
        """ValidateSummary.items() yields the new discrepancy stat fields when set."""
        from yobx.torch.validate import ValidateSummary

        s = ValidateSummary(model_id="m", prompt="p")
        s.discrepancies_max_abs = 0.005
        s.discrepancies_atol = 1e-4
        s.discrepancies_rtol = 0.1
        s.discrepancies_ratio_001 = 0.02
        s.discrepancies_ratio_01 = 0.0
        d = dict(s.items())
        self.assertAlmostEqual(d["discrepancies_max_abs"], 0.005)
        self.assertAlmostEqual(d["discrepancies_atol"], 1e-4)
        self.assertAlmostEqual(d["discrepancies_rtol"], 0.1)
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

    def test_apply_config_override_nested(self):
        """_apply_config_override forwards to text_config (multimodal)."""
        from transformers import Gemma3Config
        from yobx.torch.validate import _apply_config_override

        config = Gemma3Config()
        original_text = config.text_config.num_hidden_layers
        original_vision = config.vision_config.num_hidden_layers
        self.assertNotEqual(original_text, 2)
        _apply_config_override(config, "num_hidden_layers", 2)
        # text_config (the conventional language sub-config) receives it.
        self.assertEqual(config.text_config.num_hidden_layers, 2)
        # vision_config must NOT be touched: reducing its layer count would
        # break the conv-based vision tower (negative output sizes).
        self.assertEqual(config.vision_config.num_hidden_layers, original_vision)
        # No spurious top-level attribute is created either.
        self.assertNotIn("num_hidden_layers", vars(config))

    def test_apply_config_override_dotted_path(self):
        """_apply_config_override supports dotted paths into sub-configs."""
        from transformers import Gemma3Config
        from yobx.torch.validate import _apply_config_override

        config = Gemma3Config()
        _apply_config_override(config, "text_config.num_hidden_layers", 3)
        self.assertEqual(config.text_config.num_hidden_layers, 3)
        # Dotted form can also target the vision tower explicitly.
        _apply_config_override(config, "vision_config.num_hidden_layers", 4)
        self.assertEqual(config.vision_config.num_hidden_layers, 4)

    def test_apply_config_override_plain_attribute(self):
        """_apply_config_override sets attributes on flat (non-multimodal) configs."""
        from transformers import LlamaConfig
        from yobx.torch.validate import _apply_config_override

        config = LlamaConfig()
        _apply_config_override(config, "num_hidden_layers", 2)
        self.assertEqual(config.num_hidden_layers, 2)

    def test_load_model_uses_text_config_for_multimodal(self):
        """_load_model picks the text sub-config to instantiate a text-only causal LM."""
        import torch
        from transformers import Gemma3Config
        from yobx.torch.validate import _load_model, ValidateSummary, ValidateData

        config = Gemma3Config()
        config.text_config.num_hidden_layers = 2
        config.text_config.hidden_size = 32
        config.text_config.intermediate_size = 64
        config.text_config.num_attention_heads = 4
        config.text_config.num_key_value_heads = 2
        config.text_config.head_dim = 8

        summary = ValidateSummary(model_id="google/gemma-3-4b-it", prompt="p")
        data = ValidateData()
        model = _load_model(
            "google/gemma-3-4b-it",
            config,
            random_weights=True,
            dtype=torch.float32,
            torch_device="cpu",
            verbose=0,
            quiet=False,
            summary=summary,
            collected_data=data,
        )
        # Must be the text-only causal LM, not the full multimodal wrapper.
        self.assertEqual(type(model).__name__, "Gemma3ForCausalLM")
        self.assertFalse(hasattr(model, "vision_tower"))
        self.assertFalse(hasattr(getattr(model, "model", None), "vision_tower"))

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

    def test_validate_model_discrepancies_table_has_atol_rtol(self):
        """Each row in the discrepancies table produced by validate_model has atol and rtol columns."""  # noqa: E501
        import torch
        from yobx.torch.validate import validate_model

        custom_atol = 1e-3
        custom_rtol = 0.05
        tokenized = {
            "input_ids": torch.randint(0, 1000, (1, 5), dtype=torch.int64),
            "attention_mask": torch.ones(1, 5, dtype=torch.int64),
        }
        summary, data = validate_model(
            "arnir0/Tiny-LLM",
            tokenized_inputs=tokenized,
            random_weights=True,
            max_new_tokens=3,
            do_run=True,
            quiet=True,
            verbose=0,
            atol=custom_atol,
            rtol=custom_rtol,
        )
        self.assertIsNotNone(data.discrepancies)
        self.assertGreater(len(data.discrepancies), 0)
        for row in data.discrepancies:
            self.assertIn("atol", row, msg="atol column missing from discrepancies table row")
            self.assertIn("rtol", row, msg="rtol column missing from discrepancies table row")
            self.assertAlmostEqual(row["atol"], custom_atol)
            self.assertAlmostEqual(row["rtol"], custom_rtol)
        # The summary should also record the tolerances used.
        self.assertAlmostEqual(summary.discrepancies_atol, custom_atol)
        self.assertAlmostEqual(summary.discrepancies_rtol, custom_rtol)

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

    def test_cmd_validate_tiny_llm_patch_yobx(self):
        """End-to-end CLI invocation equivalent to:

        ``python -m yobx validate -m arnir0/Tiny-LLM -e yobx
        --opt default+onnxruntime --opset 22 --device cpu --dtype float16
        --patch yobx -r -o dump_test -v 1 --random-weights
        --config-override num_hidden_layers=2``
        """
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO
        from yobx._command_lines_parser import main

        with tempfile.TemporaryDirectory() as dump_folder:
            argv = [
                "validate",
                "-m",
                "arnir0/Tiny-LLM",
                "-e",
                "yobx",
                "--opt",
                "default+onnxruntime",
                "--opset",
                "22",
                "--device",
                "cpu",
                "--dtype",
                "float16",
                "--patch",
                "yobx",
                "-r",
                "-o",
                dump_folder,
                "-v",
                "1",
                "--random-weights",
                "--config-override",
                "num_hidden_layers=2",
            ]
            st = StringIO()
            with redirect_stdout(st):
                main(argv)
            text = st.getvalue()
        # The CLI prints a ":key,value;" summary block; sanity-check it ran.
        self.assertIn("-- summary --", text)
        self.assertIn(":model_id,arnir0/Tiny-LLM;", text)

    def test_validate_model_gemma_cli_equivalent(self):
        """Python API equivalent of the CLI command:

        ``python -m yobx validate -m google/gemma-3-4b-it -e yobx --opt default
        --opset 22 --device cpu --dtype float32 --patch all -r -o dump_test -v 1
        --random-weights --config-override num_hidden_layers=2``

        The model is gated on the HuggingFace Hub, so the test is skipped when
        the config download fails (no network, no token, or gated access).
        """
        import tempfile
        from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

        from yobx.torch.validate import validate_model, ValidateSummary, ValidateData

        http_errors: tuple = (HfHubHTTPError, OfflineModeIsEnabled)
        try:
            from requests.exceptions import ConnectionError as RequestsConnectionError
            from requests.exceptions import HTTPError, Timeout

            http_errors = (*http_errors, HTTPError, RequestsConnectionError, Timeout)
        except ImportError:
            pass
        try:
            with tempfile.TemporaryDirectory() as dump_folder:
                summary, data = validate_model(
                    model_id="google/gemma-3-4b-it",
                    exporter="yobx",
                    optimization="default",
                    opset=22,
                    device="cpu",
                    dtype="float32",
                    patch=True,
                    do_run=True,
                    dump_folder=dump_folder,
                    verbose=1,
                    random_weights=True,
                    config_overrides={"num_hidden_layers": 2},
                    quiet=True,
                )
        except http_errors as e:  # gated repo, no network, missing token, ...
            raise unittest.SkipTest(  # noqa: B904
                f"cannot validate google/gemma-3-4b-it: {type(e).__name__}: {e}"
            )

        self.assertIsInstance(summary, ValidateSummary)
        self.assertIsInstance(data, ValidateData)
        self.assertEqual(summary.model_id, "google/gemma-3-4b-it")

    @requires_torch()
    @requires_transformers("4.0")
    def test_detect_task_image_classification(self):
        """_detect_task returns 'image-classification' for vision classifiers."""
        from transformers import BeitConfig, LlamaConfig
        from yobx.torch.validate import _detect_task

        self.assertEqual(_detect_task(BeitConfig()), "image-classification")
        self.assertEqual(_detect_task(LlamaConfig()), "causal-lm")

    @requires_torch()
    @requires_transformers("4.0")
    @skipif_ci_windows("file paths")
    def test_validate_model_image_classification_random_weights(self):
        """validate_model exports a tiny Beit image classifier end-to-end."""
        import os
        import tempfile

        from transformers import BeitConfig, BeitForImageClassification
        from yobx.torch.validate import validate_model, ValidateSummary, ValidateData

        with tempfile.TemporaryDirectory() as tmp:
            config = BeitConfig(
                num_hidden_layers=2,
                hidden_size=32,
                num_attention_heads=2,
                intermediate_size=37,
                image_size=30,
                patch_size=2,
            )
            BeitForImageClassification(config).save_pretrained(tmp)

            summary, data = validate_model(
                tmp,
                random_weights=True,
                do_run=False,
                dump_folder=os.path.join(tmp, "dump"),
                verbose=0,
            )

        self.assertIsInstance(summary, ValidateSummary)
        self.assertIsInstance(data, ValidateData)
        self.assertEqual(summary.export, "OK")
        self.assertIsNotNone(data.observer)
        self.assertIsNotNone(data.kwargs)
        self.assertIn("pixel_values", data.kwargs)
        self.assertIsNotNone(data.filename)
        self.assertIsNone(summary.error_tokenizer)

    @requires_torch()
    @requires_transformers("4.0")
    def test_detect_task_feature_extraction(self):
        """_detect_task returns 'feature-extraction' for bare encoder models (e.g. Funnel)."""
        from transformers import FunnelConfig, LlamaConfig
        from yobx.torch.validate import _detect_task

        c = FunnelConfig()
        c.architectures = ["FunnelBaseModel"]
        self.assertEqual(_detect_task(c), "feature-extraction")
        c.architectures = ["FunnelModel"]
        self.assertEqual(_detect_task(c), "feature-extraction")
        # Task-specific heads keep the default causal-lm fallback (not a base model).
        c.architectures = ["FunnelForMaskedLM"]
        self.assertEqual(_detect_task(c), "causal-lm")
        self.assertEqual(_detect_task(LlamaConfig()), "causal-lm")

    @requires_torch()
    @requires_transformers("4.0")
    @skipif_ci_windows("file paths")
    def test_validate_model_funnel_base_random_weights(self):
        """validate_model routes a tiny ``FunnelBaseModel`` through feature-extraction.

        Mirrors ``python -m yobx validate -m funnel-transformer/small-base ...`` with a
        locally instantiated tiny model so the test does not need network access.

        The downstream ONNX export may itself fail on funnel attention shapes;
        the test only asserts that the validate routing correctly detects the
        feature-extraction task, loads the model via ``AutoModel`` and captures
        inputs (i.e. ``summary.n_captured >= 1`` and no tokenizer/model error).
        """
        import os

        import torch
        from transformers import FunnelConfig, FunnelBaseModel
        from yobx.torch.validate import validate_model, ValidateSummary, ValidateData

        tmp = self.get_dump_folder("test_validate_model_funnel_base_random_weights")
        config = FunnelConfig(
            vocab_size=64,
            block_sizes=[1, 1, 1],
            d_model=32,
            n_head=2,
            d_head=16,
            d_inner=32,
            max_position_embeddings=64,
        )
        config.architectures = ["FunnelBaseModel"]
        FunnelBaseModel(config).save_pretrained(tmp)

        tokenized = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
        }
        summary, data = validate_model(
            tmp,
            random_weights=True,
            do_run=False,
            dump_folder=os.path.join(tmp, "dump"),
            verbose=0,
            tokenized_inputs=tokenized,
            quiet=True,
        )

        self.assertIsInstance(summary, ValidateSummary)
        self.assertIsInstance(data, ValidateData)
        self.assertIsNone(summary.error_tokenizer)
        self.assertIsNone(summary.error_model)
        self.assertGreaterEqual(summary.n_captured or 0, 1)
        self.assertIsNotNone(data.observer)
        self.assertIsNotNone(data.kwargs)
        self.assertIn("input_ids", data.kwargs)
        self.clean_dump()


if __name__ == "__main__":
    unittest.main(verbosity=2)
