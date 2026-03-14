"""
Utilities for extracting and running ``.. runpython::`` and ``.. gdot::``
examples embedded in RST documentation files or Python docstrings.
"""

import contextlib
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Dict, List, Optional, Tuple

#: Sphinx directives whose body is executable Python code.
_EXECUTABLE_DIRECTIVES = ("runpython", "gdot")


def extract_runpython_blocks(filename: str) -> List[Dict[str, Any]]:
    """
    Extracts all ``.. runpython::`` and ``.. gdot::`` code blocks from an RST
    or Python file.

    :param filename: path to the file to parse
    :return: list of dicts, each with keys

        * ``filename`` - path to the source file
        * ``lineno`` - 1-based line number of the directive
        * ``directive`` - directive name, e.g. ``'runpython'`` or ``'gdot'``
        * ``code`` - dedented source code string ready to execute
        * ``options`` - dict of directive options (e.g. ``{'showcode': ''}``),
          or ``{'exception': ''}`` when the example is expected to raise

    .. runpython::
        :showcode:

        import os
        from yobx.helpers._check_runpython import extract_runpython_blocks

        here = os.path.dirname(os.path.abspath(__file__))
        rst = os.path.join(here, "..", "..", "docs", "cmds", "run_doc_examples.rst")
        if os.path.exists(rst):
            blocks = extract_runpython_blocks(rst)
            print(f"Found {len(blocks)} runpython/gdot block(s) in run_doc_examples.rst")
    """
    with open(filename, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    directive_pattern = re.compile(r"^(\s*)\.\.\s+(" + "|".join(_EXECUTABLE_DIRECTIVES) + r")::")

    blocks: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = directive_pattern.match(line)
        if not m:
            i += 1
            continue

        directive_indent = len(m.group(1))
        directive_name = m.group(2)
        lineno = i + 1
        options: Dict[str, str] = {}
        i += 1

        # Collect option lines (`:option:` or `:option: value`) that directly
        # follow the directive before the first blank line.
        while i < len(lines):
            stripped = lines[i].rstrip("\n\r")
            if not stripped.strip():
                # blank line - end of options section; skip it and move on
                i += 1
                break
            opt_m = re.match(r"^\s+:([\w-]+):\s*(.*)", lines[i])
            if opt_m:
                opt_m_i = re.match(r"^(\s*)", lines[i])
                if opt_m_i and len(opt_m_i.group(1)) > directive_indent:
                    options[opt_m.group(1)] = opt_m.group(2).strip()
                    i += 1
                else:
                    break
            else:
                # Not an option line and not blank - code starts immediately
                break

        # Collect code lines.  A line belongs to the block when it is blank OR
        # when its indentation is strictly greater than the directive's.
        code_lines: List[str] = []
        while i < len(lines):
            raw = lines[i].rstrip("\n\r")
            if not raw.strip():
                # blank line inside the block
                code_lines.append("")
                i += 1
                continue
            indent = len(raw) - len(raw.lstrip())
            if indent <= directive_indent:
                break
            code_lines.append(raw)
            i += 1

        # Strip trailing blank lines
        while code_lines and not code_lines[-1]:
            code_lines.pop()

        if not code_lines:
            continue

        # Dedent: remove the common leading whitespace shared by all non-blank lines
        min_indent = min(len(ln) - len(ln.lstrip()) for ln in code_lines if ln)
        dedented = [ln[min_indent:] for ln in code_lines]
        code = "\n".join(dedented) + "\n"

        blocks.append(
            {
                "filename": filename,
                "lineno": lineno,
                "directive": directive_name,
                "code": code,
                "options": options,
            }
        )

    return blocks


def run_runpython_blocks(
    files: List[str],
    verbose: int = 0,
    raise_on_error: bool = False,
    timeout: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Runs all ``.. runpython::`` and ``.. gdot::`` code blocks found in *files*.

    Each block is executed in a fresh subprocess so that failures are isolated.

    :param files: list of RST or Python file paths
    :param verbose: verbosity level (0 = silent except failures,
        1 = print each block status, 2 = also print code)
    :param raise_on_error: raise :class:`AssertionError` after processing all
        files when at least one block failed
    :param timeout: per-block timeout in seconds (``None`` means no limit)
    :return: tuple ``(n_passed, n_failed)``
    """
    n_passed = 0
    n_failed = 0

    for filepath in files:
        if not os.path.isfile(filepath):
            if verbose:
                print(f"[run-doc-examples] WARNING: file not found: {filepath!r}")
            continue

        try:
            blocks = extract_runpython_blocks(filepath)
        except Exception as exc:
            if verbose:
                print(f"[run-doc-examples] ERROR parsing {filepath!r}: {exc}")
            n_failed += 1
            continue

        for i_b, block in enumerate(blocks):
            label = f"{block['filename']}:{block['lineno']}"

            if verbose >= 2:
                print(f"\n[run-doc-examples] --- {label} ({block['directive']}) ---")
                print(textwrap.indent(block["code"], "    "))

            # Write code to a temporary file and run it
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(block["code"])
                tmp_path = tmp.name

            try:
                result = subprocess.run(
                    [sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout
                )
                raised = result.returncode != 0

                if "exception" in block["options"]:
                    # Block is expected to raise; treat it as a pass only if it
                    # actually did raise (non-zero exit), and as a failure if it
                    # exited cleanly (exception was not raised as expected).
                    success = raised
                else:
                    success = not raised

                if success:
                    n_passed += 1
                    if verbose >= 1:
                        print(f"[run-doc-examples] PASS  {i_b+1: d}/{len(blocks)} {label}")
                else:
                    n_failed += 1
                    if "exception" in block["options"] and not raised:
                        print(
                            f"[run-doc-examples] FAIL  {label}"
                            " (expected an exception but none was raised)"
                        )
                    else:
                        print(f"[run-doc-examples] FAIL  {i_b+1: d}/{len(blocks)} {label}")
                    if result.stdout:
                        print(textwrap.indent(result.stdout.rstrip(), "    stdout: "))
                    if result.stderr:
                        print(textwrap.indent(result.stderr.rstrip(), "    stderr: "))

            except subprocess.TimeoutExpired:
                n_failed += 1
                print(f"[run-doc-examples] TIMEOUT  {label}")
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    if verbose >= 1:
        total = n_passed + n_failed
        print(f"\n[run-doc-examples] {n_passed}/{total} passed, {n_failed} failed.")

    if raise_on_error and n_failed:
        raise AssertionError(
            f"{n_failed} runpython/gdot block(s) failed. "
            "Re-run with -v 2 to see the code of each block."
        )

    return n_passed, n_failed
