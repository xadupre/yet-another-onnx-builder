import unittest
from yobx.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_transformers,
)


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
        from yobx.torch.validate import validate_model

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
        self.assertIn("model_id", summary)
        self.assertEqual(summary["model_id"], "arnir0/Tiny-LLM")
        self.assertIn("export", summary)
        self.assertEqual(summary["export"], "OK")
        self.assertIn("observer", data)
        self.assertIn("kwargs", data)
        self.assertIn("dynamic_shapes", data)
        self.assertIn("filename", data)

    def test_validate_model_captures_inputs(self):
        import torch
        from yobx.torch.validate import validate_model

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
        observer = data["observer"]
        # The observer should have captured at least one input set
        self.assertGreater(len(observer.info), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
