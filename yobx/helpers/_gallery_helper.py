"""
Utilities for converting sphinx-gallery Python example files to RST.
"""

import re
import textwrap
from typing import List, Tuple


def gallery_to_rst(source: str) -> str:
    """
    Converts a sphinx-gallery Python source file to RST.

    A sphinx-gallery file has the following structure:

    * A module docstring at the very top (between ``\"\"\"`` delimiters)
      whose content is verbatim RST — title, description, labels, etc.
    * Python code interspersed with *cell separators*: a line that matches
      ``# %%`` (optionally followed by a title).
    * After a ``# %%`` separator, consecutive lines starting with ``#``
      (a comment block) are treated as RST prose.
    * All other lines (including blank lines between comment blocks and code)
      are Python code rendered as a ``.. code-block:: python`` directive.

    :param source: full text of a sphinx-gallery ``.py`` file
    :return: RST string

    .. runpython::
        :showcode:

        from yobx.helpers._gallery_helper import gallery_to_rst

        src = '''\"\"\"
        .. _l-example:

        My Example
        ==========

        A short description.
        \"\"\"

        import numpy as np

        # %%
        # Section One
        # ------------
        #
        # Some text here.

        x = np.array([1, 2, 3])
        print(x)
        '''
        print(gallery_to_rst(src))
    """
    lines = source.splitlines()
    rst_parts: List[str] = []

    # ------------------------------------------------------------------
    # 1. Extract the leading module docstring.
    # ------------------------------------------------------------------
    i = 0
    # Skip blank lines before the docstring
    while i < len(lines) and not lines[i].strip():
        i += 1

    module_doc = ""
    if i < len(lines) and lines[i].strip().startswith('"""'):
        # Find the closing triple-quote
        rest_of_line = lines[i].strip()[3:]  # content after opening """
        if rest_of_line.endswith('"""') and len(rest_of_line) > 3:
            # Single-line docstring: """text"""
            module_doc = rest_of_line[:-3]
            i += 1
        elif rest_of_line == '"""':
            # Empty single-line: skip
            i += 1
        else:
            # Multi-line: search for closing """
            doc_lines = [rest_of_line] if rest_of_line else []
            i += 1
            while i < len(lines):
                raw = lines[i]
                stripped = raw.strip()
                if '"""' in stripped:
                    # Closing delimiter found
                    before = raw[: raw.index('"""')]
                    doc_lines.append(before.rstrip())
                    i += 1
                    break
                doc_lines.append(raw)
                i += 1
            # Remove common leading indentation from doc lines
            non_empty = [ln for ln in doc_lines if ln.strip()]
            if non_empty:
                min_indent = min(len(ln) - len(ln.lstrip()) for ln in non_empty)
                doc_lines = [ln[min_indent:] if ln.strip() else "" for ln in doc_lines]
            # Strip trailing blank lines
            while doc_lines and not doc_lines[-1].strip():
                doc_lines.pop()
            module_doc = "\n".join(doc_lines)

        if module_doc.strip():
            rst_parts.append(module_doc.strip())

    # ------------------------------------------------------------------
    # 2. Parse the rest of the file into cells.
    # ------------------------------------------------------------------
    # Each cell is a list of lines that ends just before the next # %% line.
    cells: List[List[str]] = []
    current: List[str] = []

    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*#\s*%%", line):
            # Start a new cell
            if current:
                cells.append(current)
            current = [line]
        else:
            current.append(line)
        i += 1

    if current:
        # There may be code before the first # %% — treat it as an implicit cell
        cells.append(current)

    # If no # %% was found and current has content, it's all one implicit code cell
    for cell in cells:
        rst_parts.append(_cell_to_rst(cell))

    return "\n\n".join(part for part in rst_parts if part.strip()) + "\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CELL_SEP_RE = re.compile(r"^\s*#\s*%%\s*(.*)")


def _cell_to_rst(cell_lines: List[str]) -> str:
    """Convert a single cell (list of source lines) to RST."""
    if not cell_lines:
        return ""

    parts: List[str] = []

    # Check if the cell starts with a # %% separator
    first = cell_lines[0]
    m = _CELL_SEP_RE.match(first)
    inline_title = ""
    start = 0
    if m:
        inline_title = m.group(1).strip()
        start = 1

    # Split the remainder into leading comment block and trailing code block
    rst_lines, code_lines = _split_comment_code(cell_lines[start:])

    # Emit inline title if present (and not already in the comment block heading)
    if inline_title:
        # Only emit the inline title if there is no comment block or it doesn't
        # already start with the same text.
        comment_text = "\n".join(rst_lines).strip()
        if not comment_text.startswith(inline_title):
            parts.append(f"{inline_title}\n{'~' * len(inline_title)}")

    if rst_lines:
        parts.append("\n".join(rst_lines).strip())

    if code_lines:
        # Strip leading and trailing blank lines from code
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        code_text = "\n".join(code_lines)
        if code_text.strip():
            parts.append(f".. code-block:: python\n\n{textwrap.indent(code_text, '    ')}")

    return "\n\n".join(p for p in parts if p.strip())


def _split_comment_code(lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Split *lines* into a leading comment block and a trailing code block.

    Lines that start with ``#`` (after stripping leading spaces) belong to
    the comment block.  The first non-comment, non-blank line (or a blank
    line not followed by another comment line) ends the comment block.

    Comment lines have the leading ``# `` (or just ``#``) stripped.
    """
    # Find the boundary: leading run of comment/blank lines followed by the
    # first line that is neither a comment nor blank.
    comment_lines: List[str] = []
    code_lines: List[str] = []

    # Collect the leading comment block
    j = 0
    while j < len(lines):
        raw = lines[j]
        stripped = raw.strip()
        if stripped.startswith("#"):
            # Comment line: strip leading "# " or "#"
            comment_lines.append(_strip_comment_marker(raw))
            j += 1
        elif not stripped:
            # Blank line: look ahead to decide whether we're still in the
            # comment block (next non-blank line is a comment) or in code.
            lookahead = j + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead < len(lines) and lines[lookahead].strip().startswith("#"):
                comment_lines.append("")
                j += 1
            else:
                break
        else:
            break

    code_lines = lines[j:]
    return comment_lines, code_lines


def _strip_comment_marker(line: str) -> str:
    """Strip the ``#`` comment marker (and one optional space) from a line."""
    stripped = line.lstrip()
    if stripped.startswith("# "):
        return line[: len(line) - len(stripped)] + stripped[2:]
    if stripped.startswith("#"):
        return line[: len(line) - len(stripped)] + stripped[1:]
    return line
