"""Tests binary-only wheel selection without downloading or building packages."""

import os
import pathlib
import runpy
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.tags import Tag


class TestNativeWheelSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = (
            pathlib.Path(__file__).resolve().parents[2] / ".github/scripts/install_onnx_light.py"
        )
        installer = runpy.run_path(str(script))
        cls.select_wheel = staticmethod(installer["select_wheel"])
        cls.read_requirement = staticmethod(installer["read_requirement"])
        cls.release_version = staticmethod(installer["release_version"])

    def asset(self, filename):
        """Creates a release asset descriptor."""
        return {"name": filename, "browser_download_url": f"https://example.com/{filename}"}

    def test_default_version(self):
        """Keeps the wheel installer aligned with the minimum release."""
        requirement = self.read_requirement()
        self.assertIn(self.release_version(requirement), requirement.specifier)
        with tempfile.TemporaryDirectory() as directory:
            project_path = pathlib.Path(directory) / "pyproject.toml"
            for constraint in (">=1.2.3", "==2.3.4", ">=3.4.5,<4"):
                with self.subTest(constraint=constraint):
                    project_path.write_text(
                        f'[project]\ndependencies = ["numpy", "onnx-light{constraint}"]\n',
                        encoding="utf-8",
                    )
                    requirement = self.read_requirement(project_path)
                    self.assertEqual(
                        str(requirement), str(Requirement(f"onnx-light{constraint}"))
                    )
                    self.assertEqual(
                        self.release_version(requirement), constraint[2:].split(",")[0]
                    )

    def test_invalid_requirement(self):
        """Rejects missing, duplicate, or ambiguous release requirements."""
        with tempfile.TemporaryDirectory() as directory:
            project_path = pathlib.Path(directory) / "pyproject.toml"
            for dependencies in ('["numpy"]', '["onnx-light>=1", "onnx-light>=2"]'):
                with self.subTest(dependencies=dependencies):
                    project_path.write_text(
                        f"[project]\ndependencies = {dependencies}\n", encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "exactly one"):
                        self.read_requirement(project_path)
        for constraint in (">1.2.3", "<2", "==1.*", ">=2,<1"):
            with (
                self.subTest(constraint=constraint),
                self.assertRaisesRegex(ValueError, "inclusive minimum"),
            ):
                self.release_version(Requirement(f"onnx-light{constraint}"))

    def test_check_installed(self):
        """Checks installed metadata against the project requirement without downloading."""
        script = (
            pathlib.Path(__file__).resolve().parents[2] / ".github/scripts/install_onnx_light.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            metadata = pathlib.Path(directory) / "onnx_light-1.0.dist-info"
            metadata.mkdir()
            for version, succeeds in (("999.0", True), ("0.0.1", False)):
                with self.subTest(version=version):
                    (metadata / "METADATA").write_text(
                        f"Metadata-Version: 2.1\nName: onnx-light\nVersion: {version}\n",
                        encoding="utf-8",
                    )
                    result = subprocess.run(
                        [sys.executable, str(script), "--check-installed"],
                        cwd=directory,
                        env={**os.environ, "PYTHONPATH": directory},
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode == 0, succeeds, result.stderr)
                    if not succeeds:
                        self.assertIn("does not satisfy", result.stderr)

    def test_supported_versions(self):
        """Accepts the minimum release and newer versions."""
        root = pathlib.Path(__file__).resolve().parents[2]
        project = tomllib.loads((root / "pyproject.toml").read_text())
        requirements = [
            Requirement(dependency)
            for dependency in project["project"]["dependencies"]
            if Requirement(dependency).name == "onnx-light"
        ]
        self.assertEqual(len(requirements), 1)
        for version in ("0.2.5", "0.2.5.1", "0.2.6", "0.3.0"):
            with self.subTest(version=version):
                self.assertIn(version, requirements[0].specifier)
        for version in (
            "0.1.34",
            "0.2.0",
            "0.2.1",
            "0.2.2",
            "0.2.3",
            "0.2.4",
            "0.2.4.1",
            "0.2.4.2",
        ):
            with self.subTest(version=version):
                self.assertNotIn(version, requirements[0].specifier)

    def test_full_wheel_only(self):
        """Excludes reduced builds and source archives."""
        filename = "onnx_light-0.1.27-cp312-cp312-manylinux_2_28_x86_64.whl"
        assets = [
            self.asset("onnx_light-0.1.27.tar.gz"),
            self.asset("onnx_light-0.1.27-0reduced-cp312-cp312-manylinux_2_28_x86_64.whl"),
            self.asset(filename),
        ]
        self.assertEqual(
            self.select_wheel(assets, [Tag("cp312", "cp312", "manylinux_2_28_x86_64")]),
            f"https://example.com/{filename}",
        )

    def test_platform_preference(self):
        """Respects interpreter tag priority across compatible wheels."""
        filenames = [
            "onnx_light-0.1.27-cp312-cp312-manylinux_2_27_x86_64.whl",
            "onnx_light-0.1.27-cp312-cp312-manylinux_2_28_x86_64.whl",
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
                [self.asset("onnx_light-0.1.27-cp313-cp313-win_amd64.whl")],
                [Tag("cp312", "cp312", "manylinux_2_28_x86_64")],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
