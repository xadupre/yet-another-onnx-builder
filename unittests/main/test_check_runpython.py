import os
import sys
import tempfile
import unittest
from yobx.ext_test_case import ExtTestCase
from yobx.helpers._check_runpython import extract_runpython_blocks, run_runpython_blocks


class TestCheckRunpython(ExtTestCase):
    """Tests for the runpython block extractor and runner."""

    def _write_tmp(self, content: str, suffix: str = ".rst") -> str:
        """Write *content* to a temporary file and return its path."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        self.addCleanup(os.unlink, path)
        return path

    # ------------------------------------------------------------------
    # extract_runpython_blocks
    # ------------------------------------------------------------------

    def test_extract_simple_rst(self):
        content = """\
Some text.

.. runpython::
    :showcode:

    x = 1 + 1
    print(x)

More text.
"""
        path = self._write_tmp(content)
        blocks = extract_runpython_blocks(path)
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block["lineno"], 3)
        self.assertEqual(block["options"], {"showcode": ""})
        self.assertIn("x = 1 + 1", block["code"])
        self.assertIn("print(x)", block["code"])

    def test_extract_multiple_blocks_rst(self):
        content = """\
.. runpython::

    print("first")

.. runpython::

    print("second")
"""
        path = self._write_tmp(content)
        blocks = extract_runpython_blocks(path)
        self.assertEqual(len(blocks), 2)
        self.assertIn("first", blocks[0]["code"])
        self.assertIn("second", blocks[1]["code"])

    def test_extract_python_docstring(self):
        content = '''\
def foo():
    """
    Example:

    .. runpython::
        :showcode:

        import os
        print(os.getcwd())
    """
    pass
'''
        path = self._write_tmp(content, suffix=".py")
        blocks = extract_runpython_blocks(path)
        self.assertEqual(len(blocks), 1)
        self.assertIn("import os", blocks[0]["code"])
        self.assertIn("print(os.getcwd())", blocks[0]["code"])

    def test_extract_no_blocks(self):
        content = "Just plain RST without any runpython blocks.\n"
        path = self._write_tmp(content)
        blocks = extract_runpython_blocks(path)
        self.assertEqual(blocks, [])

    def test_extract_option_exception(self):
        content = """\
.. runpython::
    :exception:

    raise ValueError("expected")
"""
        path = self._write_tmp(content)
        blocks = extract_runpython_blocks(path)
        self.assertEqual(len(blocks), 1)
        self.assertIn("exception", blocks[0]["options"])

    def test_extract_code_dedented(self):
        """Code extracted from a docstring should be fully dedented."""
        content = '''\
class Foo:
    """
    .. runpython::

        x = 42
        print(x)
    """
'''
        path = self._write_tmp(content, suffix=".py")
        blocks = extract_runpython_blocks(path)
        self.assertEqual(len(blocks), 1)
        # No leading spaces in the dedented code
        first_line = blocks[0]["code"].splitlines()[0]
        self.assertFalse(first_line.startswith(" "), first_line)

    def test_extract_directive_field(self):
        """Each block dict should carry the directive name."""
        content = """\
.. runpython::

    print("rp")

.. gdot::
    :script: DOT-SECTION

    print("DOT-SECTION digraph G {}")
"""
        path = self._write_tmp(content)
        blocks = extract_runpython_blocks(path)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["directive"], "runpython")
        self.assertEqual(blocks[1]["directive"], "gdot")

    def test_extract_gdot_block(self):
        """.. gdot:: blocks are extracted just like runpython blocks."""
        content = """\
Some text.

.. gdot::
    :script: DOT-SECTION

    print("DOT-SECTION", "digraph G { A -> B }")

More text.
"""
        path = self._write_tmp(content)
        blocks = extract_runpython_blocks(path)
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block["directive"], "gdot")
        self.assertEqual(block["options"], {"script": "DOT-SECTION"})
        self.assertIn("DOT-SECTION", block["code"])

    def test_run_gdot_block(self):
        """gdot blocks are executed and contribute to pass/fail counts."""
        content = """\
.. gdot::
    :script: DOT-SECTION

    print("DOT-SECTION", "digraph G { A -> B }")
"""
        path = self._write_tmp(content)
        passed, failed = run_runpython_blocks([path], verbose=0, timeout=30)
        self.assertEqual(passed, 1)
        self.assertEqual(failed, 0)

    def test_run_gdot_and_runpython_mixed(self):
        """Files containing both directive types are handled correctly."""
        content = """\
.. runpython::

    print("hello")

.. gdot::
    :script: DOT-SECTION

    print("DOT-SECTION", "digraph G { A -> B }")
"""
        path = self._write_tmp(content)
        passed, failed = run_runpython_blocks([path], verbose=0, timeout=30)
        self.assertEqual(passed, 2)
        self.assertEqual(failed, 0)

    # ------------------------------------------------------------------
    # run_runpython_blocks
    # ------------------------------------------------------------------

    def test_run_passing_block(self):
        content = """\
.. runpython::

    print("hello from runpython")
"""
        path = self._write_tmp(content)
        passed, failed = run_runpython_blocks([path], verbose=0, timeout=30)
        self.assertEqual(passed, 1)
        self.assertEqual(failed, 0)

    def test_run_failing_block(self):
        content = """\
.. runpython::

    raise RuntimeError("intentional failure")
"""
        path = self._write_tmp(content)
        passed, failed = run_runpython_blocks([path], verbose=0, timeout=30)
        self.assertEqual(passed, 0)
        self.assertEqual(failed, 1)

    def test_run_exception_option_passes(self):
        """Blocks with :exception: are expected to raise; they should be reported as PASS."""
        content = """\
.. runpython::
    :exception:

    raise ValueError("expected")
"""
        path = self._write_tmp(content)
        passed, failed = run_runpython_blocks([path], verbose=0, timeout=30)
        self.assertEqual(passed, 1)
        self.assertEqual(failed, 0)

    def test_run_exception_option_no_raise_fails(self):
        """Blocks with :exception: that do NOT raise should be reported as FAIL."""
        content = """\
.. runpython::
    :exception:

    x = 1 + 1
"""
        path = self._write_tmp(content)
        passed, failed = run_runpython_blocks([path], verbose=0, timeout=30)
        self.assertEqual(passed, 0)
        self.assertEqual(failed, 1)

    def test_run_nonexistent_file(self):
        passed, failed = run_runpython_blocks(["/nonexistent/path.rst"], verbose=0)
        self.assertEqual(passed, 0)
        self.assertEqual(failed, 0)  # warning only, not counted as failure

    def test_run_no_blocks(self):
        content = "No runpython blocks here.\n"
        path = self._write_tmp(content)
        passed, failed = run_runpython_blocks([path], verbose=0, timeout=30)
        self.assertEqual(passed, 0)
        self.assertEqual(failed, 0)

    def test_raise_on_error(self):
        content = """\
.. runpython::

    raise RuntimeError("fail")
"""
        path = self._write_tmp(content)
        with self.assertRaises(AssertionError):
            run_runpython_blocks([path], verbose=0, raise_on_error=True, timeout=30)


class TestCheckRunpythonCLI(ExtTestCase):
    """Integration tests for the run-doc-examples CLI command."""

    def test_parser_help(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from yobx._command_lines_parser import get_parser_run_doc_examples

        st = StringIO()
        with redirect_stdout(st):
            get_parser_run_doc_examples().print_help()
        text = st.getvalue()
        self.assertIn("--timeout", text)
        self.assertIn("--ext", text)

    def test_cmd_passing(self):
        """End-to-end: _cmd_run_doc_examples must succeed on a passing file."""
        import tempfile
        import os
        from yobx._command_lines_parser import _cmd_run_doc_examples

        content = """\
.. runpython::

    print("ok")
"""
        fd, path = tempfile.mkstemp(suffix=".rst")
        os.close(fd)
        try:
            with open(path, "w") as fh:
                fh.write(content)
            # Should not raise SystemExit(1)
            _cmd_run_doc_examples(["run-doc-examples", path])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
