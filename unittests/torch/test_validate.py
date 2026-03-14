import socket
import unittest
from yobx.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_transformers,
)


def _has_network() -> bool:
    """Returns True when we can reach huggingface.co."""
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("huggingface.co", 443))
        return True
    except Exception:
        return False


@requires_torch("2.0")
@requires_transformers("5.0")
class TestValidateModel(ExtTestCase):
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

    @unittest.skipUnless(_has_network(), "No network access — skipping HuggingFace download.")
    def test_validate_model_tiny_llm(self):
        from yobx.torch.validate import validate_model

        summary, data = validate_model(
            "arnir0/Tiny-LLM",
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

    @unittest.skipUnless(_has_network(), "No network access — skipping HuggingFace download.")
    def test_validate_model_captures_inputs(self):
        from yobx.torch.validate import validate_model

        summary, data = validate_model(
            "arnir0/Tiny-LLM",
            max_new_tokens=3,
            do_run=False,
            verbose=0,
        )
        observer = data["observer"]
        # The observer should have captured at least one input set
        self.assertGreater(len(observer.info), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
