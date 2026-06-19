"""Tests for the onnx shim (USE_OPTIM_ONNX switching)."""

import importlib
import os
import sys
import types
import unittest


class TestOnnxShim(unittest.TestCase):
    """Tests that the _onnx_shim module behaves correctly."""

    def _reload_shim(self, env_value: str | None) -> types.ModuleType:
        """Reloads yobx._onnx_shim with the given USE_OPTIM_ONNX value."""
        # Remove cached module so the top-level if/else re-runs on import.
        sys.modules.pop("yobx._onnx_shim", None)
        original = os.environ.get("USE_OPTIM_ONNX")
        try:
            if env_value is None:
                os.environ.pop("USE_OPTIM_ONNX", None)
            else:
                os.environ["USE_OPTIM_ONNX"] = env_value
            return importlib.import_module("yobx._onnx_shim")
        finally:
            if original is None:
                os.environ.pop("USE_OPTIM_ONNX", None)
            else:
                os.environ["USE_OPTIM_ONNX"] = original
            # Restore cached module to the version loaded without the env var.
            sys.modules.pop("yobx._onnx_shim", None)

    def test_shim_module_importable(self):
        """Verifies that yobx._onnx_shim can be imported without errors."""
        mod = importlib.import_module("yobx._onnx_shim")
        self.assertTrue(hasattr(mod, "onnx"))

    def test_no_env_var_exposes_standard_onnx(self):
        """Verifies that the shim exposes the standard onnx module by default."""
        import onnx as real_onnx

        shim = self._reload_shim(None)
        self.assertIs(shim.onnx, real_onnx)

    def test_env_var_0_exposes_standard_onnx(self):
        """Verifies that USE_OPTIM_ONNX=0 uses standard onnx."""
        import onnx as real_onnx

        shim = self._reload_shim("0")
        self.assertIs(shim.onnx, real_onnx)

    def test_env_var_1_without_onnx_light_raises(self):
        """Verifies that USE_OPTIM_ONNX=1 without onnx_light raises ModuleNotFoundError."""
        # Only run when onnx_light is not installed.
        try:
            importlib.import_module("onnx_light.onnx")
            self.skipTest("onnx_light is installed; skipping absence test")
        except ModuleNotFoundError:
            pass

        with self.assertRaises(ModuleNotFoundError):
            self._reload_shim("1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
