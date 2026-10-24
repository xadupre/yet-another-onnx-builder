"""Tests release version consistency without importing optional dependencies."""

import ast
import pathlib
import tomllib
import unittest


class TestVersion(unittest.TestCase):
    def test_release_version(self):
        """Checks that packaging and package versions match the release."""
        root = pathlib.Path(__file__).resolve().parents[2]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["version"], "0.2.0")

        module = ast.parse((root / "yobx" / "__init__.py").read_text(encoding="utf-8"))
        versions = [
            ast.literal_eval(statement.value)
            for statement in module.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in statement.targets
            )
        ]
        self.assertEqual(versions, [project["project"]["version"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
