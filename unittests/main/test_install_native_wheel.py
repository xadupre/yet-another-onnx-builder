"""Tests binary-only wheel selection without downloading or building packages."""

import pathlib
import runpy
import unittest
from pip._vendor.packaging.tags import Tag


class TestNativeWheelSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = (
            pathlib.Path(__file__).resolve().parents[2] / ".github/scripts/install_onnx_light.py"
        )
        cls.select_wheel = staticmethod(runpy.run_path(str(script))["select_wheel"])

    def asset(self, filename):
        """Creates a release asset descriptor."""
        return {"name": filename, "browser_download_url": f"https://example.com/{filename}"}

    def test_full_wheel_only(self):
        """Excludes reduced builds and source archives."""
        filename = "onnx_light-0.1.24-cp312-cp312-manylinux_2_28_x86_64.whl"
        assets = [
            self.asset("onnx_light-0.1.24.tar.gz"),
            self.asset("onnx_light-0.1.24-0reduced-cp312-cp312-manylinux_2_28_x86_64.whl"),
            self.asset(filename),
        ]
        self.assertEqual(
            self.select_wheel(assets, [Tag("cp312", "cp312", "manylinux_2_28_x86_64")]),
            f"https://example.com/{filename}",
        )

    def test_platform_preference(self):
        """Respects interpreter tag priority across compatible wheels."""
        filenames = [
            "onnx_light-0.1.24-cp312-cp312-manylinux_2_27_x86_64.whl",
            "onnx_light-0.1.24-cp312-cp312-manylinux_2_28_x86_64.whl",
        ]
        self.assertEqual(
            self.select_wheel(
                [self.asset(name) for name in filenames],
                [
                    Tag("cp312", "cp312", "manylinux_2_28_x86_64"),
                    Tag("cp312", "cp312", "manylinux_2_27_x86_64"),
                ],
            ),
            f"https://example.com/{filenames[1]}",
        )

    def test_no_source_fallback(self):
        """Fails when the release has no matching binary."""
        with self.assertRaisesRegex(RuntimeError, "no full wheel"):
            self.select_wheel(
                [self.asset("onnx_light-0.1.24-cp313-cp313-win_amd64.whl")],
                [Tag("cp312", "cp312", "manylinux_2_28_x86_64")],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
