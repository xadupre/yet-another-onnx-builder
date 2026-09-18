"""Tests callback adaptation without importing a reference-dependent converter."""

import types
import unittest
from unittest.mock import patch
from yobx.sklearn.skl2onnx_converter import mock_guess_proto_type, patch_skl2onnx_functions


class TestConverterNamespace(unittest.TestCase):
    def test_namespace_is_restored(self):
        """Restores callback helpers on both successful and failed conversion."""
        module = types.ModuleType("native_converter_test")
        original = lambda value: value
        module.guess_proto_type = original

        def converter():
            pass

        converter.__module__ = module.__name__
        with patch.dict("sys.modules", {module.__name__: module}):
            with patch_skl2onnx_functions(converter):
                self.assertIs(module.guess_proto_type, mock_guess_proto_type)
            self.assertIs(module.guess_proto_type, original)
            with (
                self.assertRaisesRegex(RuntimeError, "conversion failed"),
                patch_skl2onnx_functions(converter),
            ):
                raise RuntimeError("conversion failed")
            self.assertIs(module.guess_proto_type, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
