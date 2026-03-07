.. _l-design-ci:

=================================
Code and CI Design (per Library)
=================================

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
     - ``yobx/xbuilder/``, ``yobx/xoptim/``,
       ``yobx/xshape/``, ``yobx/container/``,
       ``yobx/helpers/``, ``yobx/reference/``,
       ``yobx/translate/``
     - ``unittests/`` (excluding library sub-directories)
     - ``tests.yml``, ``coverage.yml``,
       ``tests_no_options.yml``
   * - scikit-learn
     - ``yobx/sklearn/``
     - ``unittests/sklearn/``
     - ``sklearn_tests.yml``, ``tests.yml``
   * - PyTorch / Transformers
     - ``yobx/torch/``
     - ``unittests/torch/``
     - ``tests.yml``,
       ``torch_transformers_tests.yml``
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

**Source layout**

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

**Tests**

Core tests live directly under ``unittests/`` (excluding the
framework-specific sub-directories):

.. code-block:: text

    unittests/
    ├── builder/
    ├── container/
    ├── helpers/
    ├── main/
    ├── reference/
    ├── translate/
    ├── xbuilder/
    ├── xoptim/
    └── xshape/

**CI workflows**

``tests.yml``
    Main multi-platform workflow (Ubuntu, macOS, Windows).
    Tests Python 3.10, 3.11, and 3.12 with both a stable release
    (torch 2.10) and the nightly PyTorch build.
    All ``unittests/tensorflow`` tests are excluded here; they run
    in the dedicated TensorFlow workflow instead.

``coverage.yml``
    Runs on Ubuntu only with Python 3.13 and nightly PyTorch.
    Generates an XML coverage report and uploads it to Codecov with
    the flag ``coverage``.
    This job provides the baseline combined coverage report.

``tests_no_options.yml``
    Installs only the core package (no ``[sklearn]``, ``[torch]``, or
    ``[tensorflow]`` extras) and runs a curated subset of tests to
    verify that the core API works without optional dependencies.

scikit-learn
============

**Source layout**

.. code-block:: text

    yobx/sklearn/
    ├── __init__.py            # re-exports to_onnx, register_sklearn_converters
    ├── convert.py             # entry-point: to_onnx()
    ├── register.py            # converter registry
    ├── sklearn_helper.py      # shared sklearn utilities
    ├── compose/               # ColumnTransformer, FeatureUnion
    ├── linear_model/          # LinearRegression, LogisticRegression, …
    ├── multiclass/            # OneVsRestClassifier, MultiOutputClassifier
    ├── neural_network/        # MLPClassifier, MLPRegressor
    ├── pipeline/              # Pipeline
    ├── preprocessing/         # StandardScaler, MinMaxScaler, …
    └── tree/                  # DecisionTreeClassifier / Regressor

Each sub-directory contains one converter module per estimator class, all
registered via :func:`register_sklearn_converters
<yobx.sklearn.register_sklearn_converters>` at first use.  See
:ref:`l-design-sklearn-converter` for the detailed converter architecture.

**Tests**

.. code-block:: text

    unittests/sklearn/
    ├── lienar_model/
    │   └── test_sklearn_linear_models.py
    ├── multiclass/
    │   └── test_sklearn_one_vs_rest.py
    ├── neural_networks/
    │   └── test_sklearn_neural_network.py
    ├── preprocessing/
    │   └── test_min_max_scaler.py
    ├── test_copilot_draft.py
    ├── test_sklearn_column_transformer.py
    ├── test_sklearn_converters.py
    └── test_sklearn_using_sklearn_onnx.py

**CI workflows**

``sklearn_tests.yml``
    Runs on Ubuntu with Python 3.12 against **two scikit-learn versions**
    (currently 1.4 and 1.8) to catch regressions introduced by minor
    API changes in scikit-learn.  Coverage is uploaded to Codecov with
    the flag ``sklearn`` for the latest version.

``tests.yml``
    The main multi-platform workflow also installs the ``[sklearn]``
    extra and therefore exercises the sklearn converters as part of the
    broad test run.

.. note::

   The ``[sklearn]`` extra also pulls in ``skl2onnx`` which is used
   as a fallback backend for the MLP converter.

PyTorch / Transformers
======================

**Source layout**

.. code-block:: text

    yobx/torch/
    ├── __init__.py
    ├── export_options.py      # ExportOptions dataclass
    ├── flatten.py             # flattening of nested inputs/outputs
    ├── flatten_helper.py
    ├── input_observer.py      # observes shapes for dynamic-axes export
    ├── patch.py               # apply / revert model patches
    ├── torch_helper.py
    ├── fake_tensor_helper.py
    ├── tiny_models.py         # small reference models used by tests
    ├── tracing.py             # core tracing / export logic
    ├── in_torch/              # patches for PyTorch internals
    └── in_transformers/       # patches for HuggingFace Transformers
        └── models/            # per-model-family patch modules

The key entry-point is ``yobx.torch.export_to_onnx`` (backed by
``tracing.py``) which calls :func:`torch.export.export`, applies the
necessary patches (see :ref:`l-design-patches`), and then converts the
resulting :class:`torch.fx.Graph` to ONNX via
:class:`GraphBuilder <yobx.xbuilder.GraphBuilder>`.

**Tests**

.. code-block:: text

    unittests/torch/

These tests cover model flattening, input observation, patches, and
the end-to-end export pipeline.

**CI workflows**

``tests.yml``
    Installs the ``[torch]`` extra and runs the full test suite
    (excluding ``unittests/tensorflow``).
    Uses a matrix of OS × Python version × torch version
    (stable 2.10 and nightly).

``torch_transformers_tests.yml``
    Dedicated workflow that installs multiple versions of
    ``transformers`` (4.49, 4.55, 4.57, and 5.2 at the time of
    writing) sequentially on Ubuntu with Python 3.13 and torch 2.10.
    This matrix verifies that the patches in ``yobx/torch/in_transformers/``
    remain compatible with each released version of the
    HuggingFace ``transformers`` library.

.. note::

   HuggingFace ``transformers`` is an optional dependency.  Tests that
   require it are guarded by a version check so that the base ``[torch]``
   install still passes without it.

TensorFlow
==========

**Source layout**

.. code-block:: text

    yobx/tensorflow/
    ├── __init__.py
    ├── convert.py             # entry-point: to_onnx()
    ├── register.py            # operator registry
    ├── tensorflow_helper.py
    └── ops/                   # one module per TF op-type
        ├── __init__.py
        ├── activations.py
        ├── bias_add.py
        ├── const.py
        └── matmul.py

The converter traces a TensorFlow concrete function via
``get_concrete_function()``, then walks the captured graph dispatching
each TF operator to its corresponding converter in ``ops/``.
All converters share the same signature:

.. code-block:: python

    def convert_<op>(g, sts, outputs, op, verbose=0): ...

where ``g`` is a :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>`,
``sts`` is a dict of symbolic type/shape info, ``outputs`` the expected
result names, and ``op`` is the TF node being translated.

**Tests**

.. code-block:: text

    unittests/tensorflow/

TensorFlow tests are **excluded** from ``tests.yml`` and
``tests_no_options.yml`` (via ``--ignore=unittests/tensorflow``) so
that the main workflow does not require a TensorFlow installation.

**CI workflows**

``tensorflow_tests.yml``
    Runs on Ubuntu with Python 3.13 and a pinned TensorFlow release
    (currently 2.20).  A coverage report is uploaded to Codecov with
    the flag ``tensorflow``.

.. note::

   TensorFlow is not listed as a top-level optional extra in
   ``pyproject.toml`` because it ships its own CUDA / CPU runtime
   binaries and must be installed separately from the rest of the
   dependencies.
