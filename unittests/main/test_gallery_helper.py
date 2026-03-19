import os
import tempfile
import textwrap
import unittest
from yobx.ext_test_case import ExtTestCase
from yobx.helpers._gallery_helper import gallery_to_rst


class TestGalleryHelper(ExtTestCase):
    """Tests for gallery_to_rst."""

    # ------------------------------------------------------------------
    # Basic docstring extraction
    # ------------------------------------------------------------------

    def test_module_docstring_only(self):
        src = textwrap.dedent('''\
            """
            .. _l-example:

            My Title
            ========

            A short description.
            """
        ''')
        rst = gallery_to_rst(src)
        self.assertIn(".. _l-example:", rst)
        self.assertIn("My Title", rst)
        self.assertIn("A short description.", rst)

    def test_single_line_docstring(self):
        src = '"""Just a one-liner."""\n\nx = 1\n'
        rst = gallery_to_rst(src)
        self.assertIn("Just a one-liner.", rst)
        self.assertIn(".. code-block:: python", rst)
        self.assertIn("x = 1", rst)

    # ------------------------------------------------------------------
    # Code cells
    # ------------------------------------------------------------------

    def test_code_after_docstring(self):
        src = textwrap.dedent('''\
            """Title
            =========
            """

            import numpy as np

            x = np.array([1, 2, 3])
        ''')
        rst = gallery_to_rst(src)
        self.assertIn(".. code-block:: python", rst)
        self.assertIn("import numpy as np", rst)
        self.assertIn("x = np.array([1, 2, 3])", rst)

    def test_cell_separator_with_title(self):
        src = textwrap.dedent('''\
            """Title
            =========
            """

            # %%
            # Section One
            # ------------
            #
            # Some text.

            x = 1
        ''')
        rst = gallery_to_rst(src)
        self.assertIn("Section One", rst)
        self.assertIn("Some text.", rst)
        self.assertIn("x = 1", rst)

    def test_cell_separator_inline_title(self):
        src = textwrap.dedent('''\
            """Title
            =========
            """

            # %% My inline title

            x = 42
        ''')
        rst = gallery_to_rst(src)
        self.assertIn("My inline title", rst)
        self.assertIn("x = 42", rst)

    def test_multiple_cells(self):
        src = textwrap.dedent('''\
            """Title
            =========
            """

            import os

            # %%
            # First section

            a = 1

            # %%
            # Second section

            b = 2
        ''')
        rst = gallery_to_rst(src)
        self.assertIn("First section", rst)
        self.assertIn("Second section", rst)
        self.assertIn("a = 1", rst)
        self.assertIn("b = 2", rst)

    def test_code_only_no_docstring(self):
        src = "x = 1\ny = 2\n"
        rst = gallery_to_rst(src)
        self.assertIn(".. code-block:: python", rst)
        self.assertIn("x = 1", rst)

    def test_comment_only_cell(self):
        """A cell with RST prose but no code should not emit a code-block."""
        src = textwrap.dedent('''\
            """Title
            =========
            """

            # %%
            # Just prose here.
        ''')
        rst = gallery_to_rst(src)
        self.assertIn("Just prose here.", rst)
        # No Python code so no code-block should appear
        self.assertNotIn(".. code-block:: python", rst)

    def test_real_gallery_example(self):
        """Smoke-test with an actual gallery file from the repository."""
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        example = os.path.join(repo_root, "docs", "examples", "core", "plot_dot_graph.py")
        if not os.path.isfile(example):
            self.skipTest("gallery example not found")
        with open(example, encoding="utf-8") as fh:
            src = fh.read()
        rst = gallery_to_rst(src)
        self.assertIn(".. _l-plot-dot-graph:", rst)
        self.assertIn("to_dot", rst)
        self.assertIn(".. code-block:: python", rst)
        self.assertIn("Build a small model", rst)

    def test_output_ends_with_newline(self):
        src = textwrap.dedent('''\
            """Title
            =========
            """
            x = 1
        ''')
        rst = gallery_to_rst(src)
        self.assertTrue(rst.endswith("\n"), f"RST output should end with newline, got: {rst!r}")

    def test_write_to_file(self):
        """gallery_to_rst result can be written to and read back from a file."""
        src = textwrap.dedent('''\
            """My gallery example
            ======================
            """

            # %%
            # Step 1

            x = 1
        ''')
        rst = gallery_to_rst(src)
        fd, path = tempfile.mkstemp(suffix=".rst")
        try:
            os.close(fd)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(rst)
            with open(path, encoding="utf-8") as fh:
                read_back = fh.read()
            self.assertEqual(rst, read_back)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
