"""
Tests that no Python source file in :mod:`yobx` is missing from the API
documentation under ``docs/api/``.
"""

import os
import re
import unittest

from yobx import __file__ as yobx_file
from yobx.ext_test_case import ExtTestCase

ROOT = os.path.realpath(os.path.abspath(os.path.join(yobx_file, "..", "..")))


def _collect_py_modules(yobx_dir: str) -> set:
    """Return a set of all dotted module names found in *yobx_dir*."""
    modules = set()
    for root, dirs, files in os.walk(yobx_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, os.path.dirname(yobx_dir))
            if fname == "__init__.py":
                mod = os.path.dirname(rel).replace(os.sep, ".")
            else:
                mod = rel.replace(os.sep, ".")[:-3]  # strip .py
            modules.add(mod)
    return modules


def _collect_doc_modules(docs_api_dir: str) -> set:
    """Return all module names *covered* by the API docs.

    A module is considered covered if:

    * it is directly referenced by any ``.. auto*::`` directive, **or**
    * it is an ancestor package of such a reference (e.g. an index file that
      contains ``.. autofunction:: yobx.pkg.foo`` implicitly covers
      ``yobx.pkg`` and ``yobx``).
    """
    auto_re = re.compile(r"\.\. auto\w+::\s+([\w.]+)")
    covered = set()
    for root, _dirs, files in os.walk(docs_api_dir):
        for fname in files:
            if not fname.endswith(".rst"):
                continue
            with open(os.path.join(root, fname), encoding="utf-8") as fh:
                for line in fh:
                    m = auto_re.search(line)
                    if m:
                        ref = m.group(1)
                        covered.add(ref)
                        # Every ancestor package is also considered covered.
                        parts = ref.split(".")
                        for i in range(1, len(parts)):
                            covered.add(".".join(parts[:i]))
    return covered


def _is_private(module: str) -> bool:
    """Return True if any dotted component of *module* begins with ``_``."""
    return any(part.startswith("_") for part in module.split("."))


# Modules that are intentionally absent from ``docs/api/`` coverage.
# Each entry below must include a brief comment explaining why it is excluded.
_KNOWN_EXCLUSIONS = frozenset(
    {
        # Internal ONNX extended-op kernel implementations.
        # Their public symbols are re-exported through yobx.reference.ops,
        # which is itself documented via automodule.
        "yobx.reference.ops.op__extended_add_add_mul_mul",
        "yobx.reference.ops.op__extended_mul_sigmoid",
        "yobx.reference.ops.op__extended_negxplus1",
        "yobx.reference.ops.op__extended_replace_zero",
        "yobx.reference.ops.op__extended_rotary",
        "yobx.reference.ops.op__extended_scatternd_of_shape",
        "yobx.reference.ops.op__extended_transpose_cast",
        "yobx.reference.ops.op__extended_tri_matrix",
        "yobx.reference.ops.op_attention",
        "yobx.reference.ops.op_bias_softmax",
        "yobx.reference.ops.op_complex",
        "yobx.reference.ops.op_fused_matmul",
        "yobx.reference.ops.op_memcpy_host",
        "yobx.reference.ops.op_qlinear_average_pool",
        "yobx.reference.ops.op_qlinear_conv",
        "yobx.reference.ops.op_quick_gelu",
        "yobx.reference.ops.op_simplified_layer_normalization",
        "yobx.reference.ops.op_skip_layer_normalization",
        # Internal torch-based op implementations.
        # Public API is re-exported through yobx.reference.torch_ops (__init__),
        # which is documented via automodule.
        "yobx.reference.torch_ops.access_ops",
        "yobx.reference.torch_ops.binary_ops",
        "yobx.reference.torch_ops.controlflow_ops",
        "yobx.reference.torch_ops.generator_ops",
        "yobx.reference.torch_ops.nn_ops",
        "yobx.reference.torch_ops.other_ops",
        "yobx.reference.torch_ops.reduce_ops",
        "yobx.reference.torch_ops.sequence_ops",
        "yobx.reference.torch_ops.shape_ops",
        "yobx.reference.torch_ops.unary_ops",
        # xshape.expressions is an internal subpackage whose symbols are all
        # re-exported at the yobx.xshape level (e.g. yobx.xshape.evaluate_expressions).
        # The top-level forwarding modules are individually documented.
        "yobx.xshape.expressions",
        "yobx.xshape.expressions.evaluate_expressions",
        "yobx.xshape.expressions.expressions_torch",
        "yobx.xshape.expressions.rename_expressions",
        "yobx.xshape.expressions.simplify_expressions",
    }
)


class TestApiDocumentation(ExtTestCase):
    def test_no_file_missing_in_doc(self):
        """Every non-private public module in yobx/ must have an automodule
        entry in docs/api/, or appear in the known exclusion list."""
        yobx_dir = os.path.join(ROOT, "yobx")
        docs_api_dir = os.path.join(ROOT, "docs", "api")

        py_modules = _collect_py_modules(yobx_dir)
        doc_modules = _collect_doc_modules(docs_api_dir)

        # Filter: private modules are allowed to be undocumented.
        public_modules = {m for m in py_modules if not _is_private(m)}

        missing = public_modules - doc_modules - _KNOWN_EXCLUSIONS
        if missing:
            lines = "\n".join(f"  - {m}" for m in sorted(missing))
            self.fail(
                f"The following modules have no ``.. automodule::`` entry in "
                f"docs/api/ and are not in the known exclusion list.\n"
                f"Add an automodule entry to the appropriate RST file, or add "
                f"the module to _KNOWN_EXCLUSIONS in this test with a reason.\n\n"
                f"{lines}"
            )

    def test_no_stale_exclusion(self):
        """Every entry in _KNOWN_EXCLUSIONS must correspond to an actual file in yobx/.

        This prevents the exclusion list from silently accumulating stale entries
        when modules are renamed or removed.
        """
        yobx_dir = os.path.join(ROOT, "yobx")
        py_modules = _collect_py_modules(yobx_dir)

        stale = _KNOWN_EXCLUSIONS - py_modules
        if stale:
            lines = "\n".join(f"  - {m}" for m in sorted(stale))
            self.fail(
                f"The following entries in _KNOWN_EXCLUSIONS no longer correspond "
                f"to any Python source file in yobx/.\n"
                f"Please remove them from the exclusion list.\n\n"
                f"{lines}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
