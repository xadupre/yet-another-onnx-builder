import unittest
import os
import sys
import subprocess
import time
import textwrap
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock
from yobx import __file__ as yobx_file
from yobx.ext_test_case import (
    ExtTestCase,
    requires_sklearn,
    is_windows,
    ignore_errors,
    has_ipython,
    has_jax,
    has_sklearn,
    has_sksurv,
    has_spox,
    has_tensorflow,
    has_torch,
    has_transformers,
)

VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(yobx_file, "..", "..")))


class TestDocumentationExamples(ExtTestCase):
    def run_rst_examples(self, relative_path):
        """Executes embedded documentation examples and returns their output."""
        path = Path(ROOT) / "docs" / relative_path
        lines = path.read_text(encoding="utf-8").splitlines()
        executed = 0
        output = StringIO()
        for index, line in enumerate(lines):
            if line != ".. runpython::":
                continue
            end = index + 1
            while end < len(lines) and (not lines[end].strip() or lines[end].startswith("    ")):
                end += 1
            block = lines[index + 1 : end]
            while block and (not block[0].strip() or block[0].lstrip().startswith(":")):
                block.pop(0)
            source = textwrap.dedent("\n".join(block))
            with self.subTest(line=index + 1), redirect_stdout(output):
                exec(compile(source, f"{path}:{index + 1}", "exec"), {})
            executed += 1
        self.assertGreater(executed, 0)
        return output.getvalue()

    def test_graph_builder_rst_examples(self):
        output = self.run_rst_examples("design/builder/graph_builder.rst")
        self.assertIn("initializer shape: (64, 32)", output)

    @requires_sklearn()
    def test_native_builder_protocol_rst_examples(self):
        output = self.run_rst_examples("design/misc/graph_builder_protocol.rst")
        self.assertIn("output: name='probabilities'", output)

    @requires_sklearn()
    def test_expected_api_rst_examples(self):
        output = self.run_rst_examples("design/sklearn/expected_api.rst")
        self.assertIn("Sub(X,", output)

    def run_test(self, fold: str, name: str, verbose=0) -> int:
        ppath = os.environ.get("PYTHONPATH", "")
        if not ppath:
            os.environ["PYTHONPATH"] = ROOT
        elif ROOT not in ppath:
            sep = ";" if is_windows() else ":"
            os.environ["PYTHONPATH"] = ppath + sep + ROOT
        perf = time.perf_counter()
        cmds = [sys.executable, "-u", os.path.join(fold, name)]
        env = dict(os.environ, MPLBACKEND="Agg")
        p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        _out, err = p.communicate()
        st = err.decode("utf-8", errors="replace")
        if p.returncode:
            if '"dot" not found in path.' in st:
                raise unittest.SkipTest(f"failed: {name!r} due to missing dot.")
            if (
                "We couldn't connect to 'https://huggingface.co'" in st
                or "Cannot access content at: https://huggingface.co/" in st
            ):
                raise unittest.SkipTest(f"Connectivity issues due to\n{err}")
            raise AssertionError(
                f"Example {name!r} (cmd: {cmds} - exec_prefix={sys.exec_prefix!r}) "
                f"failed with exit code {p.returncode} due to\n{st}"
            )
        dt = time.perf_counter() - perf
        if verbose:
            print(f"{dt:.3f}: run {name!r}")
        return 1

    def test_subprocess_failure_without_traceback(self):
        process = mock.Mock(returncode=1)
        process.communicate.return_value = (b"", b"Native execution failed")
        with (
            mock.patch.object(subprocess, "Popen", return_value=process) as popen,
            self.assertRaisesRegex(AssertionError, "Native execution failed"),
        ):
            self.run_test(ROOT, "native_failure.py")
        self.assertEqual(popen.call_args.kwargs["env"]["MPLBACKEND"], "Agg")

    @classmethod
    def add_test_methods(cls):
        this = os.path.abspath(os.path.dirname(__file__))
        root_fold = os.path.normpath(os.path.join(this, "..", "..", "docs", "examples"))
        # Collect (fold, name) pairs from all subdirectories
        found = []
        for subdir in ("core", "sklearn", "torch", "tensorflow"):
            fold = os.path.join(root_fold, subdir)
            if os.path.isdir(fold):
                for name in os.listdir(fold):
                    if name.endswith(".py") and name.startswith("plot_"):
                        found.append((fold, name))
        has_dot = int(os.environ.get("UNITTEST_DOT", "0"))
        for fold, name in found:
            reason = None

            if (
                not reason
                and not has_dot
                and name
                in {
                    "plot_dot_graph.py",
                    "plot_einsum.py",
                    "plot_dump_intermediate_results.py",
                    "plot_export_report.py",
                    "plot_input_observer_tiny_llm.py",
                    "plot_sklearn_convert_options.py",
                    "plot_jax_to_onnx.py",
                    "plot_sklearn_custom_converter_options.py",
                    "plot_sklearn_dataframe_pipeline.py",
                    "plot_sklearn_function_options.py",
                    "plot_sklearn_function_transformer.py",
                    "plot_sklearn_kmeans.py",
                    "plot_sklearn_pipeline.py",
                    "plot_sklearn_with_sklearn_onnx.py",
                    "plot_tensorflow_to_onnx.py",
                }
            ):
                reason = "dot not installed"

            if (
                not reason
                and name in {"plot_input_observer_transformers.py", "plot_patch_model.py"}
                and not has_transformers("4.57")
            ):
                reason = "transformers<4.57"

            if not reason and sys.platform.startswith(("win", "darwin")):
                reason = "CI complains on Windows"

            if (
                not reason
                and not has_torch()
                and name
                in {
                    "plot_evaluator_comparison.py",
                    "plot_flattening.py",
                    "plot_input_observer.py",
                    "plot_mini_onnx_builder.py",
                    "plot_input_observer_tiny_llm.py",
                    "plot_input_observer_transformers.py",
                    "plot_patch_model.py",
                }
            ):
                reason = "torch not installed"

            if (
                not reason
                and not has_transformers()
                and name in {"plot_input_observer_tiny_llm.py"}
            ):
                reason = "transformers not installed"

            if not reason and not has_sklearn() and "sklearn" in name:
                reason = "scikit-learn not installed"

            if not reason and not has_sklearn() and name in {"plot_tree_statistics.py"}:
                reason = "scikit-learn not installed"

            if not reason and not has_sklearn("1.8") and name in {"plot_sklearn_pls_float32.py"}:
                reason = "expected discrepancies with scikit-learn<1.8"

            if not reason and not has_spox() and "spox" in name:
                reason = "spox not installed"

            if not reason and not has_sksurv() and "sksurv" in name:
                reason = "scikit-survival not installed"

            if not reason and not has_tensorflow() and "tensorflow" in name:
                reason = "tensorflow not installed"

            if not reason and not has_jax() and "jax" in name:
                reason = "jax not installed"

            if not reason and not has_ipython() and "mermaid" in name:
                reason = "IPython not installed"

            if reason:

                @unittest.skip(reason)
                def _test_(self, fold=fold, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            else:

                @ignore_errors(OSError)  # connectivity issues
                def _test_(self, fold=fold, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            short_name = os.path.split(os.path.splitext(name)[0])[-1]
            setattr(cls, f"test_{short_name}", _test_)


TestDocumentationExamples.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)
