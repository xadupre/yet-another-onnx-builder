"""Tests for the onnx shim (USE_OPTIM_ONNX switching)."""

import importlib
import sys
import unittest


class TestOnnxShim(unittest.TestCase):
    """Tests that the _onnx_shim module behaves correctly."""

    def test_no_env_var_leaves_onnx_unchanged(self):
        """Verifies that the shim does not replace onnx when USE_OPTIM_ONNX is unset."""
        import onnx

        import yobx._onnx_shim  # noqa: F401

        # onnx should still be the real onnx package
        self.assertIs(sys.modules["onnx"], onnx)

    def test_shim_module_importable(self):
        """Verifies that yobx._onnx_shim can be imported without errors."""
        mod = importlib.import_module("yobx._onnx_shim")
        self.assertTrue(hasattr(mod, "_apply_onnx_light"))

    def test_env_var_1_without_onnx_light_raises(self):
        """Verifies that USE_OPTIM_ONNX=1 without onnx_light raises ImportError."""
        import os

        # Only run this sub-test when onnx_light is not actually installed.
        try:
            import onnx_light  # type: ignore[import-untyped]  # noqa: F401

            self.skipTest("onnx_light is installed; skipping absence test")
        except ModuleNotFoundError:
            pass

        original_onnx = sys.modules.get("onnx")
        # Temporarily remove the shim from sys.modules so _apply_onnx_light
        # runs again when we reimport it under the env var.
        shim_key = "yobx._onnx_shim"
        original_shim = sys.modules.pop(shim_key, None)
        try:
            os.environ["USE_OPTIM_ONNX"] = "1"
            with self.assertRaises(ImportError):
                importlib.import_module(shim_key)
        finally:
            os.environ.pop("USE_OPTIM_ONNX", None)
            if original_shim is not None:
                sys.modules[shim_key] = original_shim
            else:
                sys.modules.pop(shim_key, None)
            # Restore onnx in sys.modules
            if original_onnx is not None:
                sys.modules["onnx"] = original_onnx


if __name__ == "__main__":
    unittest.main(verbosity=2)
