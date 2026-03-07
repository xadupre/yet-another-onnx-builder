"""
Unit tests for yobx.helpers.copilot.
"""

import json
import os
import textwrap
import unittest
from unittest.mock import MagicMock, patch

from yobx.ext_test_case import ExtTestCase, requires_sklearn


@requires_sklearn("1.4")
class TestExtractPythonCode(ExtTestCase):
    """Tests for _extract_python_code."""

    def _fct(self, text: str) -> str:
        from yobx.helpers.copilot import _extract_python_code

        return _extract_python_code(text)

    def test_fenced_python_block(self):
        text = "Here is the code:\n```python\nprint('hello')\n```\nDone."
        self.assertEqual(self._fct(text), "print('hello')")

    def test_fenced_plain_block(self):
        text = "```\nprint('hello')\n```"
        self.assertEqual(self._fct(text), "print('hello')")

    def test_no_fence_returns_stripped(self):
        text = "  print('hello')  "
        self.assertEqual(self._fct(text), "print('hello')")

    def test_prefers_python_fence_over_plain(self):
        text = "```python\ncode_a\n```\n```\ncode_b\n```"
        self.assertEqual(self._fct(text), "code_a")

    def test_multiline_code(self):
        code = "import os\n\nprint(os.getcwd())"
        text = f"```python\n{code}\n```"
        self.assertEqual(self._fct(text), code)


@requires_sklearn("1.4")
class TestInferSubmodule(ExtTestCase):
    """Tests for _infer_submodule."""

    def _fct(self, cls) -> str:
        from yobx.helpers.copilot import _infer_submodule

        return _infer_submodule(cls)

    def test_linear_model(self):
        from sklearn.linear_model import LogisticRegression

        self.assertEqual(self._fct(LogisticRegression), "linear_model")

    def test_preprocessing(self):
        from sklearn.preprocessing import StandardScaler

        self.assertEqual(self._fct(StandardScaler), "preprocessing")

    def test_tree(self):
        from sklearn.tree import DecisionTreeClassifier

        self.assertEqual(self._fct(DecisionTreeClassifier), "tree")

    def test_unknown_module_falls_back(self):
        class _FakeEstimator:
            __module__ = "other.stuff"

        self.assertEqual(self._fct(_FakeEstimator), "misc")


@requires_sklearn("1.4")
class TestBuildConverterPrompt(ExtTestCase):
    """Tests for _build_converter_prompt."""

    def _fct(self, cls) -> str:
        from yobx.helpers.copilot import _build_converter_prompt

        return _build_converter_prompt(cls)

    def test_contains_class_name(self):
        from sklearn.linear_model import Ridge

        prompt = self._fct(Ridge)
        self.assertIn("Ridge", prompt)

    def test_contains_sklearn_module(self):
        from sklearn.linear_model import Ridge

        prompt = self._fct(Ridge)
        self.assertIn("sklearn", prompt)

    def test_contains_example_code(self):
        from sklearn.preprocessing import StandardScaler

        prompt = self._fct(StandardScaler)
        self.assertIn("register_sklearn_converter", prompt)

    def test_regressor_kind(self):
        from sklearn.linear_model import Ridge

        prompt = self._fct(Ridge)
        self.assertIn("regressor", prompt)

    def test_classifier_kind(self):
        from sklearn.linear_model import LogisticRegression

        prompt = self._fct(LogisticRegression)
        self.assertIn("classifier", prompt)

    def test_transformer_kind(self):
        from sklearn.preprocessing import StandardScaler

        prompt = self._fct(StandardScaler)
        self.assertIn("transformer", prompt)


@requires_sklearn("1.4")
class TestDraftConverterWithCopilot(ExtTestCase):
    """Integration tests for draft_converter_with_copilot using mocked HTTP."""

    _FAKE_CODE = textwrap.dedent("""\
        from typing import Dict, List
        from sklearn.linear_model import Ridge
        from ...typing import GraphBuilderExtendedProtocol
        from ..register import register_sklearn_converter


        @register_sklearn_converter(Ridge)
        def sklearn_ridge(
            g: GraphBuilderExtendedProtocol,
            sts: Dict,
            outputs: List[str],
            estimator: Ridge,
            X: str,
            name: str = "ridge",
        ) -> str:
            \"\"\"Converts Ridge into ONNX.\"\"\"
            assert isinstance(estimator, Ridge)
            return g.op.Gemm(X, estimator.coef_, estimator.intercept_, name=name, outputs=outputs)
        """).strip()

    def _make_fake_completions_response(self):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"```python\n{self._FAKE_CODE}\n```",
                        }
                    }
                ]
            }
        ).encode()

    def _make_fake_token_response(self):
        return json.dumps({"token": "fake-copilot-token-abc123"}).encode()

    def test_dry_run_no_file_written(self):
        """dry_run=True should return code without touching the filesystem."""
        from yobx.helpers.copilot import draft_converter_with_copilot
        from sklearn.linear_model import Ridge

        fake_token_resp = self._make_fake_token_response()
        fake_code_resp = self._make_fake_completions_response()

        call_count = [0]

        def fake_urlopen(req):
            call_count[0] += 1
            mock_resp = MagicMock()
            if "copilot_internal" in req.full_url:
                mock_resp.read.return_value = fake_token_resp
            else:
                mock_resp.read.return_value = fake_code_resp
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            code = draft_converter_with_copilot(
                Ridge,
                token="fake-github-pat",
                dry_run=True,
                verbose=0,
            )

        self.assertEqual(code, self._FAKE_CODE)
        # two HTTP calls: token exchange + completions
        self.assertEqual(call_count[0], 2)

    def test_writes_file_to_output_dir(self):
        """Without dry_run, the code should be written to output_dir."""
        import tempfile

        from yobx.helpers.copilot import draft_converter_with_copilot
        from sklearn.linear_model import Ridge

        fake_token_resp = self._make_fake_token_response()
        fake_code_resp = self._make_fake_completions_response()

        def fake_urlopen(req):
            mock_resp = MagicMock()
            if "copilot_internal" in req.full_url:
                mock_resp.read.return_value = fake_token_resp
            else:
                mock_resp.read.return_value = fake_code_resp
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with tempfile.TemporaryDirectory() as tmp:
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                code = draft_converter_with_copilot(
                    Ridge,
                    token="fake-github-pat",
                    output_dir=tmp,
                    dry_run=False,
                    verbose=0,
                )

            out_file = os.path.join(tmp, "ridge.py")
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, encoding="utf-8") as fh:
                written = fh.read().strip()
            self.assertEqual(written, self._FAKE_CODE)
            self.assertEqual(code, self._FAKE_CODE)

    def test_creates_init_py_for_new_subpackage(self):
        """When output_dir is a new (empty) directory, an __init__.py is created."""
        import tempfile

        from yobx.helpers.copilot import draft_converter_with_copilot
        from sklearn.linear_model import Ridge

        fake_token_resp = self._make_fake_token_response()
        fake_code_resp = self._make_fake_completions_response()

        def fake_urlopen(req):
            mock_resp = MagicMock()
            if "copilot_internal" in req.full_url:
                mock_resp.read.return_value = fake_token_resp
            else:
                mock_resp.read.return_value = fake_code_resp
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with tempfile.TemporaryDirectory() as tmp:
            new_pkg = os.path.join(tmp, "new_subpkg")
            # new_subpkg does NOT exist yet
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                draft_converter_with_copilot(
                    Ridge,
                    token="fake-github-pat",
                    output_dir=new_pkg,
                    dry_run=False,
                )

            self.assertTrue(os.path.exists(os.path.join(new_pkg, "__init__.py")))

    def test_raises_without_token(self):
        """ValueError should be raised when no token is available."""
        from yobx.helpers.copilot import draft_converter_with_copilot
        from sklearn.linear_model import Ridge

        # Ensure env vars are unset
        env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "GH_TOKEN")}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
            draft_converter_with_copilot(Ridge, token=None)

    def test_token_from_env_var(self):
        """Token should be read from the GITHUB_TOKEN env var when not passed."""
        from yobx.helpers.copilot import draft_converter_with_copilot
        from sklearn.linear_model import Ridge

        fake_token_resp = self._make_fake_token_response()
        fake_code_resp = self._make_fake_completions_response()

        def fake_urlopen(req):
            mock_resp = MagicMock()
            if "copilot_internal" in req.full_url:
                mock_resp.read.return_value = fake_token_resp
            else:
                mock_resp.read.return_value = fake_code_resp
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "env-fake-pat"}),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            code = draft_converter_with_copilot(Ridge, dry_run=True)
        self.assertEqual(code, self._FAKE_CODE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
