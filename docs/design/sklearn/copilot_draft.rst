.. _l-design-copilot-draft:

=======================================
Drafting Converters with GitHub Copilot
=======================================

:func:`yobx.helpers.copilot.draft_converter_with_copilot` uses the
`GitHub Copilot <https://github.com/features/copilot>`_ chat API to
generate a first-draft ONNX converter for any
:epkg:`scikit-learn` estimator and writes the result into the
``yobx/sklearn/`` sub-package tree.

High-level workflow
===================

.. code-block:: text

    GitHub PAT
         │
         ▼
    _get_copilot_token()      ← exchanges PAT for ephemeral Copilot token
         │
         ▼
    _build_converter_prompt() ← assembles context-rich prompt:
         │                       • existing StandardScaler converter example
         │                       • GraphBuilder API reference
         │                       • estimator class / module / kind
         ▼
    _call_copilot_api()       ← POST to Copilot chat-completions endpoint
         │
         ▼
    _extract_python_code()    ← strips Python source from markdown response
         │
         ▼
    write to disk             ← yobx/sklearn/<submodule>/<snake_case>.py
                                 (creates __init__.py for new sub-packages)

Usage
=====

The easiest entry point is :func:`yobx.helpers.copilot.draft_converter_with_copilot`.
It is also re-exported from :mod:`yobx.sklearn` for convenience.

Supply a GitHub personal-access token with the ``copilot`` scope, or set
the ``GITHUB_TOKEN`` / ``GH_TOKEN`` environment variable:

.. code-block:: python

    from sklearn.linear_model import Ridge
    from yobx.helpers.copilot import draft_converter_with_copilot

    # Preview the generated code without writing any file
    code = draft_converter_with_copilot(Ridge, dry_run=True)
    print(code)

    # Write to yobx/sklearn/linear_model/ridge.py
    draft_converter_with_copilot(Ridge, token="ghp_...")

After the file is written, you must:

1. Import the new module in the matching ``register()`` function
   (e.g. ``yobx/sklearn/linear_model/__init__.py``).
2. Review and test the generated converter — Copilot may need manual
   corrections.

``dry_run`` mode
================

Pass ``dry_run=True`` to print and return the code without touching the
filesystem.  This is useful for reviewing the Copilot output before
committing to disk:

.. code-block:: python

    code = draft_converter_with_copilot(Ridge, dry_run=True)

Sub-module inference
====================

:func:`~yobx.helpers.copilot._infer_submodule` maps the estimator's
Python module path (e.g. ``sklearn.linear_model._ridge``) to the target
``yobx/sklearn/`` sub-package name (e.g. ``linear_model``).  Estimators
from unknown modules fall back to ``misc``.

Prompt construction
===================

:func:`~yobx.helpers.copilot._build_converter_prompt` injects:

* The full source of the ``StandardScaler`` converter as a concrete
  example, so Copilot can mirror the exact conventions.
* A concise ``GraphBuilder`` API reference (the most commonly used
  ``g.op.*`` methods plus helpers such as ``g.set_type``,
  ``g.unique_name``, etc.).
* The estimator's class name, sklearn module, and kind
  (``classifier`` / ``regressor`` / ``transformer``).

This reduces hallucination and produces output that typically requires
only minor adjustments.

API reference
=============

See :ref:`api-copilot`. 
