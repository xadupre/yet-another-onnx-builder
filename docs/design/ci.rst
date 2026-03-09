.. _l-design-ci:

=======================================
Repository Structure and CI Workflows
=======================================

This page explains how the source code is organized and how the
continuous-integration workflows are structured for each component of
**yet-another-onnx-builder** (``yobx``).

Overview
========

The repository is organized as a set of mostly-independent optional sub-packages
under ``yobx/``.  Each sub-package has its own unit-test directory under
``unittests/`` and typically a dedicated CI workflow.
The table below summarizes the mapping:

.. list-table::
   :widths: 20 25 25 30
   :header-rows: 1

   * - Library
     - Source
     - Tests
     - CI Workflows
   * - Core / Builder
     - ``yobx/xbuilder/``, ``yobx/xoptim/``, ``yobx/xshape/``, ``yobx/container/``, ``yobx/helpers/``, ``yobx/reference/``, ``yobx/translate/``
     - ``unittests/`` (excluding library sub-directories)
     - ``core_tests.yml``
   * - scikit-learn
     - ``yobx/sklearn/``
     - ``unittests/sklearn/``
     - ``sklearn_tests.yml``
   * - PyTorch / Transformers
     - ``yobx/torch/``
     - ``unittests/torch/``
     - ``torch_tests.yml``, ``ci_transformers_dev.yml``, ``ci_transformers_releases.yml``
   * - TensorFlow
     - ``yobx/tensorflow/``
     - ``unittests/tensorflow/``
     - ``tensorflow_tests.yml``

On top of the per-library workflows there are cross-cutting quality
checks that run on every push:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Workflow
     - Purpose
   * - ``build.yml``
     - Builds and uploads the Python package (``python -m build``).
   * - ``docs.yml``
     - Builds the Sphinx documentation (triggered by ``.rst`` / ``.md``
       changes only).
   * - ``mypy.yml``
     - Static type-checking of the whole ``yobx/`` package.
   * - ``style.yml``
     - Linting with ``ruff``.
   * - ``spelling.yml``
     - Spell-checking of documentation strings with ``codespell``.
   * - ``pyrefly.yml``
     - Experimental type-analysis with ``pyrefly``.

Core / Builder
==============

The core of the library lives in several tightly-coupled sub-packages:

* ``yobx/xbuilder/`` — :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>`,
  the main API for constructing and optimizing ONNX graphs.
  See :ref:`l-design-graph-builder` for a full description.
* ``yobx/xoptim/`` — pattern-based graph optimizer and all optimization
  patterns (``patterns/``, ``patterns_ort/``, ``patterns_ml/``,
  ``patterns_exp/``).
  See :ref:`l-design-pattern-optimizer-patterns`.
* ``yobx/xshape/`` — symbolic shape-expression system used internally by
  ``GraphBuilder``.
  See :ref:`l-design-shape`.
* ``yobx/builder/`` — lightweight builder helpers and the
  :class:`OnnxScriptGraphBuilder <yobx.builder.onnxscript.OnnxScriptGraphBuilder>`
  bridge.
* ``yobx/container/`` — :class:`ModelContainer` for grouping several ONNX
  models that share initializers.
* ``yobx/helpers/`` — general ONNX and runtime utilities shared by all
  sub-packages.
* ``yobx/reference/`` — reference evaluators (``onnxruntime``, ``torch``).
* ``yobx/translate/`` — translates ONNX graphs back to executable Python.

Tests
=====

Core tests live directly under ``unittests/``. Each library has its own subfolder
and CI file.
