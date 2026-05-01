import unittest
import os
import sys
import importlib.util
import subprocess
import time
from yobx import __file__ as yobx_file
from yobx.ext_test_case import (
    ExtTestCase,
    is_windows,
    ignore_errors,
    has_jax,
    has_onnx_ir,
    has_onnx_shape_inference,
    has_sklearn,
    has_sksurv,
    has_spox,
    has_tensorflow,
    has_torch,
    has_transformers,
)

VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(yobx_file, "..", "..")))


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}'.".format(module_name, module_file_path)
        )
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExamples(ExtTestCase):
    def run_test(self, fold: str, name: str, verbose=0) -> int:
        ppath = os.environ.get("PYTHONPATH", "")
        if not ppath:
            os.environ["PYTHONPATH"] = ROOT
        elif ROOT not in ppath:
            sep = ";" if is_windows() else ":"
            os.environ["PYTHONPATH"] = ppath + sep + ROOT
        perf = time.perf_counter()
        try:
            mod = import_source(fold, os.path.splitext(name)[0])
            assert mod is not None
        except FileNotFoundError:
            # try another way
            cmds = [sys.executable, "-u", os.path.join(fold, name)]
            p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res = p.communicate()
            _out, err = res
            st = err.decode("ascii", errors="ignore")
            if st and "Traceback" in st:
                if '"dot" not found in path.' in st:
                    # dot not installed, this part
                    # is tested in onnx framework
                    raise unittest.SkipTest(f"failed: {name!r} due to missing dot.")
                if (
                    "We couldn't connect to 'https://huggingface.co'" in st
                    or "Cannot access content at: https://huggingface.co/" in st
                ):
                    raise unittest.SkipTest(f"Connectivity issues due to\n{err}")
                raise AssertionError(  # noqa: B904
                    "Example '{}' (cmd: {} - exec_prefix='{}') "
                    "failed due to\n{}"
                    "".format(name, cmds, sys.exec_prefix, st)
                )
        dt = time.perf_counter() - perf
        if verbose:
            print(f"{dt:.3f}: run {name!r}")
        return 1

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

            if (
                not reason
                and (not has_onnx_ir() or not has_onnx_shape_inference())
                and name in {"plot_computed_shapes.py"}
            ):
                reason = "onnx_ir is missing"

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
