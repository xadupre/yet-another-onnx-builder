"""Tests dataframe dependency version requirements."""

import pathlib
import tomllib
import unittest


class TestOptionalDependencies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = pathlib.Path(__file__).resolve().parents[2]
        project = tomllib.loads((root / "pyproject.toml").read_text())
        cls.extras = project["project"]["optional-dependencies"]

    def test_pandas_versions(self):
        """Requires the updated pandas release in development and documentation extras."""
        for extra in ("dev", "docs"):
            with self.subTest(extra=extra):
                self.assertEqual(
                    [dep for dep in self.extras[extra] if dep.startswith("pandas")],
                    ["pandas>=3.1.0"],
                )

    def test_polars_version(self):
        """Requires the updated polars release in its optional extra."""
        self.assertEqual(self.extras["polars"], ["polars>=2.0.0"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
