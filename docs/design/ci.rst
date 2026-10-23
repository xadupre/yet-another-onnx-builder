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
     - ``yobx/xbuilder/``, ``yobx/builder/onnxlight/``, ``yobx/xshape/``, ``yobx/container/``, ``yobx/helpers/``, ``yobx/reference/``
     - ``unittests/`` (excluding library sub-directories)
     - ``core_tests.yml``
   * - scikit-learn (and xgboost, lightgbm, perpetual)
     - ``yobx/sklearn/``
     - ``unittests/sklearn/``
     - ``ci_sklearn.yml``, ``ci_perpetual.yml``
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
       changes only).  TensorFlow gallery examples are built first in a
       dedicated single-threaded pass
       (``sphinx_gallery_conf.filename_pattern=/tensorflow/``), then the
       full documentation is built in a second pass.
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
* ``yobx/builder/onnxlight/`` — native graph builder bridge that exposes the
  published ``onnx-light`` ``GraphGraph`` optimizer through
  :class:`~yobx.xbuilder.OptimizationOptions`.
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

Tests
=====

Core tests live directly under ``unittests/``. Each library has its own subfolder
and CI file.

See :ref:`l-design-env-variables` for a reference of all environment variables
that can be set to control test execution and enable extra diagnostic output.

Native fusion coverage and remaining regressions
-----------------------------------------------

Tests previously skipped pending ``onnx-light 0.1.33`` are exercised with the
currently required ``0.2.0`` release. Nested conditional functions and dynamic
grouped matrix multiplication work with that release. The grouped matrix
multiplication test checks both retained and inlined ``Scan`` functions.
The reshape/MatMul composition tests run Cast elimination before the fusion
pass because ``0.2.0`` canonicalizes Identity nodes at optimization entry.
Tiny-LLM fusion tests use opset 24 and run without version-based expected
failures, including float16 exports and transformer patches. They check the
standard ``Attention`` and ``RMSNormalization`` operators and a retained native
rotary fusion.

Native Attention and RotaryEmbedding patterns already exist in ``0.1.32``.
The export bridge previously expanded optimizer-generated functions again
when inlining converter functions, undoing these fusions. It now inlines only
the original converter functions and retains functions introduced by native
optimization, including their nested dependencies. Shape inference expands a
copy of the model without removing fusions from the exported model.
Selective inlining uses the native graph operation to keep function opsets and
value-lifetime metadata consistent. Explicit inlining also removes discarded
imported definitions from the bridge's function registry.

Focused numerical regressions check the native rewrite reports for
``FunctionAttention``, ``FunctionHalfRotaryEmbedding`` and ``RotaryEmbedding``,
including repeated exports and nested function exports. The standard
``RotaryEmbedding`` pattern requires unshared doubled cosine/sine caches.
Tiny-LLM shares caches between query and key and may instead retain the native
``HalfRotaryEmbedding`` function.

The native regressions fixed in ``0.1.33`` run without expected failures:

* Tiny-LLM submodule export preserves unknown dimensions instead of ``INT64_MAX``
  (``xadupre/onnx-light#5298``).
* The sequence-splitting optimization emits split sizes that cover the input
  dimension (``xadupre/onnx-light#5299``). The regression also checks numerical
  outputs.

Their conditional ``pytest.mark.xfail`` markers apply only to older releases;
the required ``0.2.0`` release no longer allows these failures.
