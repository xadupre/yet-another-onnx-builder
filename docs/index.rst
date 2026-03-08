yet-another-onnx-builder documentation
======================================

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/tests.yml

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/docs.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/docs.yml

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/style.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/style.yml

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/spelling.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/spelling.yml

.. image:: https://codecov.io/gh/xadupre/yet-another-onnx-builder/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/xadupre/yet-another-onnx-builder

.. image:: https://img.shields.io/github/repo-size/xadupre/yet-another-onnx-builder
    :target: https://github.com/xadupre/yet-another-onnx-builder

`yet-another-onnx-builder on GitHub <https://github.com/xadupre/yet-another-onnx-builder>`_

**yet-another-onnx-builder** (``yobx``) is a toolkit for converting machine learning models
to `ONNX <https://onnx.ai>`_ format and manipulating ONNX graphs programmatically.
It provides:

- A :ref:`graph builder API <l-design-graph-builder>` for constructing and optimizing ONNX graphs,
  with built-in shape inference and a pattern-based
  :ref:`graph optimizer <l-design-pattern-optimizer-patterns>`.
- Converters for **scikit-learn** estimators and pipelines (``yobx.sklearn``).
- Utilities for **PyTorch** export, including model patching and input flattening (``yobx.torch``).
- A symbolic :ref:`shape expression system <l-design-shape>` for dynamic shape handling at export time.
- A :ref:`translation tool <l-design-translate>` that converts ONNX graphs back to executable Python code.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design/index
   cmds/index
   auto_examples_core/index
   auto_examples_sklearn/index
   auto_examples_torch/index
   auto_examples_tensorflow/index

.. toctree::
   :maxdepth: 2
   :caption: API

   api/index

.. toctree::
   :maxdepth: 2
   :caption: misc

   index_stats

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Older versions
==============

* `0.1.0 <../v0.1.0/index.html>`_
